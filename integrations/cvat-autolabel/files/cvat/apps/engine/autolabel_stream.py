"""Pure merge policy plus atomic CVAT persistence for progressive annotations."""
from copy import deepcopy
import hashlib
import json


class MergeConflict(ValueError):
    pass


class _Absent:
    def __deepcopy__(self,memo):
        return self


ABSENT = _Absent()


def normalize(value):
    if isinstance(value,dict):
        return {k:normalize(v) for k,v in value.items() if k not in {"clientID","keyframe"} and not (k=="id" and v is None)}
    if isinstance(value,list):
        return [normalize(v) for v in value]
    if isinstance(value,float) and value.is_integer():
        return int(value)
    return value


def signature(row):
    return hashlib.sha256(json.dumps(normalize({k:v for k,v in row.items() if k not in {"id","clientID"}}),
        sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def without_ids(value):
    if isinstance(value,dict):
        return {k:without_ids(v) for k,v in value.items() if k!="id"}
    if isinstance(value,list):
        return [without_ids(v) for v in value]
    return value


def created_client_ids(planned, persisted, submitted):
    remaining=list(submitted)
    result={}
    for row,saved in zip(planned,persisted,strict=True):
        match=next((old for old in remaining if normalize(without_ids(old))==normalize(without_ids(row))),None)
        if match is not None:
            remaining.remove(match)
            if "clientID" in match:
                result[str(match["clientID"])]=saved["id"]
    return result


def track_identity(track):
    return next((a["value"] for a in track.get("attributes",[]) if str(a.get("value","")).startswith("vm:")),None)


def geometry_signature(shape):
    return signature({k:shape.get(k) for k in ("type","frame","points","outside","rotation")})


def incremental_merge(current, state, operations):
    current,state = deepcopy(current),deepcopy(state)
    state.setdefault("tracks",{})
    creates,updates = [],[]
    for operation in operations:
        identity = operation["identity"]
        if not identity.startswith("vm:") or len(identity)>200:
            raise ValueError("Invalid progressive track identity")
        known = state["tracks"].get(identity)
        track = next((t for t in current["tracks"] if track_identity(t)==identity),None)
        if known and not track:
            known["deleted"] = True
        if known and known.get("deleted"):
            continue
        shape = {"type":"rectangle","frame":operation["frame"],"points":operation["points"],
                 "outside":False,"occluded":False,"keyframe":True,"rotation":0,"z_order":0,"attributes":operation.get("shape_attributes",[])}
        if track is None:
            track = {"frame":operation["frame"],"label_id":operation["label_id"],"group":0,"source":"auto",
                     "attributes":operation["attributes"],"shapes":[]}
            current["tracks"].append(track)
            creates.append(track)
            known = {"owned":{},"deleted":False}
            state["tracks"][identity] = known
        owned = known["owned"]
        geometry=known.setdefault("geometry",{})
        removed=set(known.get("deleted_frames",[]))
        present={str(s["frame"]) for s in track["shapes"]}
        removed.update(f for f in owned if f not in present)
        known["deleted_frames"]=sorted(removed)
        if str(shape["frame"]) in removed:
            continue
        # Human outside boundaries terminate automatic extension.
        terminated = any(s["outside"] and s["frame"]<=shape["frame"] and
            (geometry.get(str(s["frame"])) != geometry_signature(s) if str(s["frame"]) in geometry
             else owned.get(str(s["frame"])) != signature(s)) for s in track["shapes"])
        existing = next((s for s in track["shapes"] if s["frame"]==shape["frame"]),None)
        if terminated or (existing and (not existing["outside"] or
            geometry.get(str(existing["frame"]),geometry_signature(existing) if owned.get(str(existing["frame"]))==signature(existing) else None)!=geometry_signature(existing))):
            continue
        if existing:
            if owned.get(str(existing["frame"]))!=signature(existing):
                for field in ("attributes","occluded","z_order"):
                    shape[field]=deepcopy(existing[field])
            track["shapes"].remove(existing)
        track["shapes"].append(shape)
        owned[str(shape["frame"])] = signature(shape)
        geometry[str(shape["frame"])]=geometry_signature(shape)
        if shape["frame"]+1<operation["total_frames"]:
            outside = {**shape,"frame":shape["frame"]+1,"outside":True}
            if not any(s["frame"]==outside["frame"] for s in track["shapes"]):
                track["shapes"].append(outside)
                owned[str(outside["frame"])] = signature(outside)
                geometry[str(outside["frame"])]=geometry_signature(outside)
        track["shapes"].sort(key=lambda s:s["frame"])
        if track not in creates and track not in updates:
            updates.append(track)
    return current,state,creates,updates


def _three_way(base, local, remote, path, choices):
    if local==base:
        return deepcopy(remote)
    if remote==base or local==remote:
        return deepcopy(local)
    if all(isinstance(v,dict) for v in (base,local,remote)):
        out = {}
        missing = ABSENT
        for key in set(base)|set(local)|set(remote):
            values = [v.get(key,missing) for v in (base,local,remote)]
            result = _three_way(*values,f"{path}/{key}",choices)
            if result is not missing:
                out[key] = result
        return out
    if all(isinstance(v,list) for v in (base,local,remote)) and path.endswith("/shapes"):
        maps = [{str(s["frame"]):s for s in v} for v in (base,local,remote)]
        result = _three_way(*maps,path+"/frames",choices)
        return sorted(result.values(),key=lambda s:s["frame"])
    if all(isinstance(v,list) for v in (base,local,remote)) and path.endswith("/attributes"):
        maps=[{str(a["spec_id"]):a for a in v} for v in (base,local,remote)]
        result=_three_way(*maps,path+"/specs",choices)
        return sorted(result.values(),key=lambda a:a["spec_id"])
    if choices.get(path) in ("local","remote"):
        return deepcopy(local if choices[path]=="local" else remote)
    raise MergeConflict(path)


def merge_save(base, local, remote, state, choices=None):
    choices=choices or {}
    base,local,remote = normalize(base),normalize(local),normalize(remote)
    merged = deepcopy(remote)
    for kind in ("tags","shapes","tracks","intervals"):
        before = {row["id"]:row for row in base.get(kind,[]) if "id" in row}
        after = {row["id"]:row for row in local.get(kind,[]) if "id" in row}
        server = {row["id"]:row for row in remote.get(kind,[]) if "id" in row}
        for identity,old in before.items():
            if identity not in after:
                current = server.get(identity)
                if current and current!=old:
                    # Deleting a track extended only by the model is safe; another human's edits conflict.
                    known = state.get("tracks",{}).get(track_identity(current),{})
                    previous_shapes = {s["frame"]:s for s in old.get("shapes",[])}
                    auto_only = kind=="tracks" and all(current.get(k)==v for k,v in old.items() if k!="shapes") and all(
                        (s["frame"] in previous_shapes and signature(s)==signature(previous_shapes[s["frame"]])) or known.get("owned",{}).get(str(s["frame"]))==signature(s)
                        for s in current.get("shapes",[]))
                    if not auto_only:
                        path=f"{kind}/{identity}/delete"
                        if choices.get(path)=="remote":
                            continue
                        if choices.get(path)!="local":
                            raise MergeConflict(path)
                server.pop(identity,None)
            elif after[identity]!=old:
                if identity not in server:
                    path=f"{kind}/{identity}/deleted"
                    if choices.get(path)=="remote":
                        continue
                    # Restore after another user's deletion is deliberately a new human annotation.
                    if choices.get(path)=="local":
                        restored=deepcopy(after[identity])
                        restored.pop("id",None)
                        for shape in restored.get("shapes",[]):
                            shape.pop("id",None)
                        merged.setdefault("restored",{}).setdefault(kind,[]).append(restored)
                        continue
                    raise MergeConflict(path)
                server[identity] = _three_way(old,after[identity],server[identity],f"{kind}/{identity}",choices)
        created = [row for row in local.get(kind,[]) if "id" not in row]
        if any(identity not in before for identity in after):
            raise ValueError("Save contains unknown annotation IDs")
        merged[kind] = list(server.values())+created+merged.get("restored",{}).get(kind,[])
    merged.pop("restored",None)
    return merged


def handle_stream(*,job_id,data,mode,transaction,job_model,get_annotations,patch_annotations,put_annotations):
    """Called inside the same row lock as every native CVAT annotation writer."""
    from cvat.apps.engine.models import AutoLabelStreamState,AutoLabelStreamOperation
    from cvat.apps.engine.autolabel_revision import annotation_revision
    with transaction.atomic():
        job_model.objects.select_for_update().get(pk=job_id)
        row,_created = AutoLabelStreamState.objects.get_or_create(job_id=job_id,defaults={"state":{}})
        current = get_annotations(job_id)
        if mode=="save":
            operation_id=data["operation_id"]
            digest=hashlib.sha256(json.dumps({"base":data["base"],"local":data["local"]},sort_keys=True,
                separators=(",",":"),allow_nan=False).encode()).hexdigest()
            previous=AutoLabelStreamOperation.objects.filter(job_id=job_id,operation_id=operation_id).first()
            if previous:
                if previous.digest!=digest:
                    return 409,{"detail":"Save identity changed"}
                return 200,{"revision":annotation_revision(current),"annotations":current,
                            "created_ids":previous.result.get("created_ids",{}),"replayed":True}
        if mode=="incremental":
            operation_id = data["operation_id"]
            digest = signature({k:v for k,v in data.items() if k!="expected_revision"})
            previous = AutoLabelStreamOperation.objects.filter(job_id=job_id,operation_id=operation_id).first()
            if previous:
                if previous.digest!=digest:
                    return 409,{"detail":"Operation identity changed"}
                return 200,{"revision":annotation_revision(current),"replayed":True}
            if data["expected_revision"] != annotation_revision(current):
                return 409,{"detail":"CVAT annotations changed"}
            _merged,state,creates,updates = incremental_merge(current,row.state,data["operations"])
            if creates:
                patch_annotations(job_id,{"version":0,"tags":[],"shapes":[],"tracks":creates,"intervals":[]},"create")
            if updates:
                patch_annotations(job_id,{"version":0,"tags":[],"shapes":[],"tracks":updates,"intervals":[]},"update")
            persisted=get_annotations(job_id)
            planned={track_identity(t):t for t in _merged["tracks"]}
            for track in persisted["tracks"]:
                identity=track_identity(track)
                if identity not in state["tracks"] or identity not in planned:
                    continue
                desired={str(s["frame"]):s for s in planned[identity]["shapes"]}
                owned=state["tracks"][identity]["owned"]
                for shape in track["shapes"]:
                    frame=str(shape["frame"])
                    if frame in desired and owned.get(frame)==signature(desired[frame]):
                        owned[frame]=signature(shape)
            row.state = state
            row.save(update_fields=["state"])
            AutoLabelStreamOperation.objects.create(job_id=job_id,operation_id=operation_id,digest=digest)
        else:
            if (data.get("choices") or data.get("require_revision") is True) and data.get("expected_revision")!=annotation_revision(current):
                return 409,{"detail":"Conflict choices became stale","conflict":"revision",
                            "revision":annotation_revision(current),"current":current}
            try:
                merged = merge_save(data["base"],data["local"],current,row.state,data.get("choices"))
            except MergeConflict as error:
                return 409,{"detail":"Concurrent annotation edit","conflict":str(error),
                            "revision":annotation_revision(current),"current":current}
            # Delete/update/create a delta instead of replacing all rows and changing IDs.
            created_ids={}
            for kind in ("tags","shapes","tracks","intervals"):
                existing = {r["id"]:r for r in current.get(kind,[])}
                intended = {r["id"]:r for r in merged[kind] if "id" in r}
                for action,rows in (("delete",[{"id":i} for i in existing if i not in intended]),
                                    ("update",[r for i,r in intended.items() if r!=existing[i]]),
                                    ("create",[r for r in merged[kind] if "id" not in r])):
                    if rows:
                        payload = {"version":0,"tags":[],"shapes":[],"tracks":[],"intervals":[]}
                        payload[kind] = rows
                        result=patch_annotations(job_id,payload,action)
                        if action=="create":
                            created_ids[kind]=created_client_ids(rows,result[kind],data["local"].get(kind,[]))
            present = {track_identity(t) for t in get_annotations(job_id)["tracks"]}
            for identity,known in row.state.get("tracks",{}).items():
                if identity not in present:
                    known["deleted"] = True
            row.save(update_fields=["state"])
            AutoLabelStreamOperation.objects.create(job_id=job_id,operation_id=operation_id,digest=digest,
                                                   result={"created_ids":created_ids})
        updated = get_annotations(job_id)
        return 200,{"revision":annotation_revision(updated),"annotations":updated,
                    "created_ids":created_ids if mode=="save" else {}}
