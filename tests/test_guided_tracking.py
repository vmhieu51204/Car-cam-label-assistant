from pathlib import Path
import numpy as np
import pytest
from inference.sam3_guided import propagate,prompt_arguments,frame_passes


def test_box_and_point_coordinates_are_relative():
    box={'kind':'box','bbox':{'xtl':20,'ytl':10,'xbr':60,'ybr':30}}
    assert prompt_arguments(box,100,50)['box'].tolist()==pytest.approx(np.array([[.2,.2,.6,.6]]))
    result=prompt_arguments({'kind':'points','points':[{'x':20,'y':10,'label':1},{'x':80,'y':30,'label':0}]},100,50)
    assert result=={'points':[[.2,.2],[.8,.6]],'labels':[1,0]}


def test_passes_cover_both_sides_of_arbitrary_anchor():
    assert frame_passes(5,2,'both')==[[2,3,4],[2,1,0]]
    assert frame_passes(5,0,'backward')==[[0]]


class Tracker:
    def __init__(self,missing=False):self.orders=[];self.prompts=[];self.missing=missing
    def init_state(self,video_path,**kwargs):
        order=[p.read_text() for p in sorted(Path(video_path).glob('*.jpg'))]
        self.orders.append(order);return order
    def add_new_points_or_box(self,**kwargs):self.prompts.append(kwargs)
    def propagate_in_video(self,state,**kwargs):
        for i in range(len(state)-(1 if self.missing else 0)):
            mask=np.zeros((10,20));mask[2:6,3:9]=1
            yield i,[1],None,[mask],None


def test_actual_worker_maps_backwards_frames_and_preserves_human_anchor(tmp_path):
    root=tmp_path/'input';root.mkdir();frames=[]
    for f in range(5):
        (root/f'{f}.jpg').write_text(str(f));frames.append({'relative_path':f'{f}.jpg'})
    bbox={'xtl':1.,'ytl':1.,'xbr':12.,'ybr':8.}
    request={'width':20,'height':10,'frames':frames,'direction':'both','prompt':{'kind':'box','frame':2,'bbox':bbox}}
    tracker=Tracker();result=propagate(tracker,request,root,tmp_path/'work')
    assert tracker.orders==[['2','3','4'],['2','1','0']]
    assert [o['frame_index'] for o in result]==[0,1,2,3,4]
    assert result[2]['bbox']==bbox
    assert result[0]['bbox']=={'xtl':3.,'ytl':2.,'xbr':9.,'ybr':6.}
    assert all(o['score'] is None for o in result)
    with pytest.raises(ValueError,match='every requested frame'):
        propagate(Tracker(missing=True),request,root,tmp_path/'missing')
