// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import * as THREE from 'three';

function disposeMaterial(material: THREE.Material, disposed: Set<unknown>): void {
    if (disposed.has(material)) return;
    disposed.add(material);
    Object.keys(material).forEach((prop) => {
        const value = material[prop];
        if (value instanceof THREE.Texture && !disposed.has(value)) {
            disposed.add(value);
            value?.dispose();
        }
    });
    material.dispose();
}

export function disposeObjectResources(object: THREE.Object3D, disposed = new Set<unknown>()): void {
    if ('geometry' in object && object.geometry instanceof THREE.BufferGeometry && !disposed.has(object.geometry)) {
        disposed.add(object.geometry);
        object.geometry.dispose();
    }

    if ('material' in object) {
        if (Array.isArray(object.material)) {
            object.material.forEach((material) => {
                if (material instanceof THREE.Material) {
                    disposeMaterial(material, disposed);
                }
            });
        } else if (object.material instanceof THREE.Material) {
            disposeMaterial(object.material, disposed);
        }
    }

    if ('renderTarget' in object && object.renderTarget instanceof THREE.WebGLRenderTarget && !disposed.has(object.renderTarget)) {
        disposed.add(object.renderTarget);
        object.renderTarget.dispose();
    }
}

export function disposeObject3D(object: THREE.Object3D, disposed = new Set<unknown>()): void {
    while (object.children.length > 0) {
        const child = object.children[0];
        object.remove(child);
        disposeObject3D(child, disposed);
    }

    disposeObjectResources(object, disposed);
}

export function disposeScene(scene: THREE.Scene, disposed = new Set<unknown>()): void {
    while (scene.children.length > 0) {
        const child = scene.children[0];
        scene.remove(child);
        disposeObject3D(child, disposed);
    }
}
