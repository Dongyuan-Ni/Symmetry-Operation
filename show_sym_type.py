############################################################################################
# 此脚本仅适用于纯碳结构且需满足Inversion=diag(-1,-1,-1)
# 判断一个对称操作是恒等(identity)、旋转(rotation)、空间反演(inversion)、镜面反射(reflection)、
# 转动反演(rotoinversion)、螺旋轴(screw rotation)、滑移面(glide reflection)中的哪一种
# 看一个对称性R：Det(R)=1,则为旋转或螺旋轴；Det(R)=-1为空间反演、镜面反射、滑移面或转动反演
# 分辨旋转或螺旋轴可利用：原子经对称操作前后的坐标差（是一个矢量）是否与螺旋轴垂直（均在笛卡尔坐标系下）
# 分辨转动反演和滑移面的方法：先判断是否为滑移面：判断unit cell中所有原子镜面反射后的滑移矢量是否相等
# 更具体的：滑移操作下的原子经对称操作前后的坐标差（是一个矢量）可以分为垂直分量v_v（平行于镜面对称轴）和
# 水平分量v_h（垂直于镜面对称轴），每一个原子的v_h都应该相等
# 最后还有一种是转动反演(rotoinversion)
############################################################################################
import numpy as np
from numpy.linalg import *
from pymatgen import *
import spglib
import irrep.spacegroup
from pymatgen.io.cif import Structure

def convert_to_spglib_format(lat, coords, spes):
    spg_spes = list(spes)
    spg_lat = [tuple(i) for i in lat]
    spg_coords = [tuple(i) for i in coords]
    cell = (spg_lat, spg_coords, spg_spes)
    sym = spglib.get_symmetry(cell, symprec=1e-3)
    sg_number = spglib.get_symmetry_dataset(cell, symprec=1e-3, angle_tolerance=-1.0, hall_number=0)['number']
    return sym, sg_number

def judge_glide_plane(B, cart_coords, rot, trans, axis):
    v_v_tot = []
    v_h_tot = []
    for i in range(len(cart_coords)):
        coord_transform = B.dot(rot).dot(inv(B)).dot(cart_coords[i]) + B.dot(trans)
        v = coord_transform - cart_coords[i]
        n_axis = axis / norm(axis)
        v_v = v.dot(n_axis) * n_axis
        v_v_tot.append(v_v)
        v_h = v - v_v
        v_h_tot.append(v_h)
    idx = 0
    for i in range(len(v_h_tot)):
        if norm(v_h_tot[i] - v_h_tot[0]) < 1e-3:
            idx += 1
    if idx == len(v_h_tot):
        return True
    else:
        return False

def get_operation_type(lat, cart_coords, rot, trans):
# 晶格
    B = lat.T
#求对称操作本征值并排序
    rotxyz = B.dot(rot).dot(inv(B))
    E, V = eig(rotxyz)

    if E.prod() < 0:
        inversion = True
        E *= -1
    else:
        inversion = False

    idx = np.argsort(E.real)
    E = E[idx]
    V = V[:, idx]
    axis = V[:, 2].real

    ##############################
    axis_frac = inv(B).dot(axis)
    print('Axis: {}'.format(axis_frac))
    angle = np.angle(E[0], deg=True)
    print('Angle: {}'.format(angle))
    ##############################

# Det(R)==1:
    if inversion == False:
        # identity
        if np.isclose(E, 1).all():
            return 'Identity'
        else:
            coord_diff = B.dot(rot).dot(inv(B)).dot(cart_coords[0]) + B.dot(trans) - cart_coords[0]
            # rotation
            if np.isclose(coord_diff.dot(axis), 0, atol=1e-3, rtol=1e-3).all():
                return 'Rotation'
            # screw rotation
            else:
                return 'Screw Rotation'
# Det(R)==-1:
    elif inversion == True:
        # inversion
        if np.isclose(E, 1).all():
            return 'Inversion'
        else:
            coord_diff = B.dot(rot).dot(inv(B)).dot(cart_coords[0]) + B.dot(trans) - cart_coords[0]
            # reflection
            if np.isclose(np.cross(coord_diff, axis), 0, atol=1e-3, rtol=1e-3).all():
                return 'Reflection'
            # glide plane
            elif judge_glide_plane(B, cart_coords, rot, trans, axis):
                return 'Glide Plane'
            # rotoinversion
            else:
                return 'Rotoinversion'

def main():
    struct = Structure.from_file('test5.cif')
    lat = struct.lattice._matrix
    coords = struct.frac_coords
    cart_coords = struct.cart_coords
    spes = struct.atomic_numbers
    sym, sg_number = convert_to_spglib_format(lat, coords, spes)
    print('Space group: {}'.format(sg_number))
    print('{} symmetry operations in total'.format(len(sym['rotations'])))
    for i in range(len(sym['rotations'])):
        print()
        print('Num: {}'.format(i))
        print(sym['rotations'][i],sym['translations'][i])
        print(get_operation_type(lat, cart_coords, sym['rotations'][i], sym['translations'][i]))

if __name__ == '__main__':
    main()
