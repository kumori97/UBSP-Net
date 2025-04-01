import torch
from os.path import join, split, exists
import pickle as pkl
import numpy as np
from glob import glob
import trimesh
from psbody.mesh import Mesh
from lib.smpl_paths import SmplPaths
from models.volumetric_SMPL import VolumetricSMPL
from multiprocessing import Pool

# Number of points to sample from the scan
NUM_POINTS = 10000

def map_mesh_points_to_reference(pts, src, ref):
    closest_face, closest_points = src.closest_faces_and_points(pts)
    vert_ids, bary_coords = src.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))
    correspondences = (ref[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)

    return correspondences


def map_vitruvian_vertex_color(tgt_vertices, registered_smpl_mesh,
                               path_to_cols='./assets/vitruvian_cols.npy'):

    col = np.load(path_to_cols)
    vids, _ = registered_smpl_mesh.closest_vertices(tgt_vertices)
    vids = np.array(vids)
    return col[vids]

def process_file(path):
    print(path)
    name = split(path)[-1]
    smpl_dict = pkl.load(open(join(path, "parameters.pkl"), 'rb'), encoding='latin-1')
    gender = smpl_dict['gender']
    # Load smpl part labels
    with open('./assets/smpl_parts_dense.pkl', 'rb') as f:
        dat = pkl.load(f, encoding='latin-1')
    smpl_parts = np.zeros((6890, 1))
    for n, k in enumerate(dat):
        smpl_parts[dat[k]] = n

    vsmpl = VolumetricSMPL('./assets/volumetric_smpl_function_64_{}'.format(gender),
                                'cuda', gender)

    sp = SmplPaths(gender=gender)
    ref_smpl = sp.get_smpl()

    input_smpl = Mesh(filename=join(path, name + '_naked.obj'))
    input_scan = Mesh(filename=join(path, name + '.obj'))
    print(join(path, name + '.obj'))

    temp = trimesh.Trimesh(vertices=input_scan.v, faces=input_scan.f)
    points = temp.sample(NUM_POINTS)

    ind, _ = input_smpl.closest_vertices(points)
    part_labels = smpl_parts[np.array(ind)]
    correspondences = map_mesh_points_to_reference(points, input_smpl, ref_smpl.r)

    #can be viewed as the closest point that matches the external scan
    posed_scan_correspondences = vsmpl(torch.from_numpy(correspondences).to(torch.float32).cuda().unsqueeze(0),
                                            smpl_dict['pose'].cuda().unsqueeze(0),
                                            smpl_dict['betas'].cuda().unsqueeze(0),
                                            smpl_dict['trans'].cuda().unsqueeze(0))[0]
    vc = map_vitruvian_vertex_color(points, input_smpl)
    np.savetxt(join(path,'posed_scan_correspondences.txt'), posed_scan_correspondences.detach().cpu().numpy().astype('float32'))
    np.savetxt(join(path,'scan.txt'), points.astype('float32'))
    np.savetxt(join(path,'correspondences.txt'), correspondences.astype('float32'))
    np.savetxt(join(path,'part_labels.txt'), part_labels.astype('float32'))
    np.savetxt(join(path,'vc.txt'), vc.astype('float32'))


if __name__ == "__main__":
    obj_files = glob('assets/CAPE_Dataset_Sampled_10000/**/*_naked.obj', recursive=True)
    obj_files = [split(file)[0] for file in obj_files]

    # Create a pool of worker processes
    pool = Pool(processes=4)
    # Use the pool to process the files in parallel
    pool.map(process_file, obj_files)
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
    #
    # for item in obj_files:
    #     process_file(item)

