from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0.05, exposure=1.5)
scene.set_directional_light((1, 1, 1), 1, (1, 1, 1))
scene.set_floor(-60, (0.5, 0.5, 1))
scene.set_background_color((0.3, 0.4, 0.6))


@ti.func
def create_hot_ball(pos, r, h1, color, color_noise):
    temp_r = ti.round(0.2 * r)
    r2 = ti.round(ti.sqrt(r * r - temp_r * temp_r))
    for i, j, k in ti.ndrange((-r, r + 1), (-temp_r, r + 1), (-r, r + 1)):
        x = ivec3(i, j, k)
        if x.dot(x) < r * r:
            scene.set_voxel(pos + vec3(i, j, k), 1, color + (ti.random()) * 0.2)
    for i, j, k in ti.ndrange((-r2, r2 + 1), (ti.round(-temp_r - 0.7 * h1) + 2, -temp_r), (-r2, r2 + 1)):
        x = ivec3(i, 0, k)
        if x.dot(x) < ((h1 + j + temp_r) * r2 / h1) * ((h1 + j + temp_r) * r2 / h1):
            scene.set_voxel(pos + vec3(i, j, k), 1, color + (ti.random()) * 0.2)
    j = -temp_r - 0.7 * h1
    a = ti.round((h1 + j + temp_r) * r2 / h1)
    sx_min = -a
    sx_max = a
    sz_min = -a
    sz_max = a
    r3 = ti.round((sx_max - sx_min) / 2)
    h2 = ti.round(5 / 8 * r3) * 2
    c2 = ti.round(4 / 5 * r3)
    for j in ti.ndrange((ti.round(-temp_r - 0.7 * h1 - h2), ti.round(-temp_r - 0.7 * h1) + 4)):
        c = 0
        if j >= ti.round(-temp_r - 0.7 * h1):
            c = 0
        else:
            if (j - ti.round(-temp_r - 0.7 * h1 - h2)) % 2 != 0:
                c = c2 - 0.5 * ti.round((j - ti.round(-temp_r - 0.7 * h1 - h2)) / 2 - 0.2)
            else:
                c = c2 - 0.5 * (j - ti.round(-temp_r - 0.7 * h1 - h2)) / 2
        scene.set_voxel(pos + vec3(sx_min + c, j, sz_min), 1, color + (ti.random()) * 0.2)
        scene.set_voxel(pos + vec3(sx_min + c, j, sz_max), 1, color + (ti.random()) * 0.2)
        scene.set_voxel(pos + vec3(sx_max - c, j, sz_min), 1, color + (ti.random()) * 0.2)
        scene.set_voxel(pos + vec3(sx_max - c, j, sz_max), 1, color + (ti.random()) * 0.2)
    c = ti.round(4 / 5 * r3)
    h4 = 0.2 * r
    for i, j, k in ti.ndrange((sx_min + c, sx_max - c + 1),
                              (ti.round(-temp_r - 0.7 * h1 - h2) - h4, ti.round(-temp_r - 0.7 * h1 - h2)),
                              (sz_min, sz_max + 1)):
        scene.set_voxel(pos + vec3(i, j, k), 1, color + ti.random() * 0.1)
    for i, k in ti.ndrange((sx_min + c, sx_max - c + 1), (sz_min, sz_max + 1)):
        if i == sx_min + c or i == sx_max - c or k == sz_min or k == sz_max:
            scene.set_voxel(pos + vec3(i, ti.round(-temp_r - 0.7 * h1 - h2), k), 1, color + ti.random() * 0.1)


@ti.func
def create_clouds(pos, r, color):
    for I in ti.grouped(ti.ndrange((-r, r), (-r * 0.3, r * 0.3), (-r, r))):
        f = I / r
        h = 0.5 - max(f[0], -0.5) * 0.5
        d = vec2(f[1], f[2]).norm()
        prob = max(0, 1 - d) ** 2 * h
        prob *= h
        if prob < 0.15:
            prob = 0.0
        if ti.random() < prob:
            scene.set_voxel(pos + I, 1.8, color + (ti.random()) * 0.2)


@ti.kernel
def initialize_voxels():
    create_hot_ball(vec3(0, 20, 0), 10, 1.5 * 10, vec3(234 / 255, 153 / 255, 153 / 255), vec3(0.3))

    create_clouds(vec3(-12, 5, -8), 12, vec3(30 / 255, 144 / 255, 255 / 255))
    create_clouds(vec3(-2, 6, -18), 12, vec3(30 / 255, 144 / 255, 255 / 255))
    create_clouds(vec3(-8, 10, -16), 12, vec3(135 / 255, 206 / 255, 250 / 255))

    create_clouds(vec3(0, 6, -18), 12, vec3(30 / 255, 144 / 255, 255 / 255))
    create_clouds(vec3(12, 4, -16), 10, vec3(135 / 255, 206 / 255, 250 / 255))
    create_clouds(vec3(18, 2, -10), 11, vec3(30 / 255, 144 / 255, 255 / 255))
    create_clouds(vec3(9, -1, -6), 12, vec3(30 / 255, 144 / 255, 255 / 255))


initialize_voxels()
scene.finish()
