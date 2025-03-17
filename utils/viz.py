import argparse
import os
import sys
import shutil
import random
import pickle
import torch
from tqdm import tqdm
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


def plot_3d_motion_combined(save_path, kinematic_tree, gt_data, pred_data, title, figsize=(10, 10), fps=20, radius=4):
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    gt_data = gt_data.copy().reshape(len(gt_data), -1, 3)
    pred_data = pred_data.copy().reshape(len(pred_data), -1, 3)
    frame_number = min(len(gt_data), len(pred_data))  # 두 데이터 길이 중 작은 값을 기준으로 설정

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    init()
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        plot_xzPlane(
            minx=-2, 
            maxx=2, 
            miny=0, 
            minz=-2, 
            maxz=2
        )

        # GT 
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            ax.plot3D(gt_data[index, chain, 0], gt_data[index, chain, 1], gt_data[index, chain, 2], linewidth=3.0, color='red', alpha=0.5, label='GT' if i == 0 else "")

        # Prediction 
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            ax.plot3D(pred_data[index, chain, 0], pred_data[index, chain, 1], pred_data[index, chain, 2], linewidth=3.0, color='blue', alpha=0.7, label='Prediction' if i == 0 else "")

        if index == 0: 
            ax.legend(loc="upper right")

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_local(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    # ax = p3.Axes3D(fig)
    ax = p3.Axes3D(fig, auto_add_to_figure=False) 
    fig.add_axes(ax)
    init()
    MINS = data.min(axis=0).min(axis=0) # [min_x, min_y, min_z]
    MAXS = data.max(axis=0).max(axis=0) # [max_x, max_y, max_z]
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1] # min_y
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        global text_annotations
        for text in text_annotations:
            text.remove()
        text_annotations = []  # remove text in the current frame

        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        plot_xzPlane(
            minx=-2, 
            maxx=2, 
            miny=0, 
            minz=-2, 
            maxz=2
        ) # fix ground plane
        
        if index > 1:
             ax.plot3D(trajec[:index, 0]-trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1]-trajec[index, 1], linewidth=1.0,
                      color='blue')
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        # show minimum y vertex
        min_y_index = np.argmin(data[index, :, 1])  
        min_y_vertex = data[index, min_y_index]    
        ax.scatter(min_y_vertex[0], min_y_vertex[1], min_y_vertex[2], color='yellow', s=50, label='Lowest Y')

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    ani.save(save_path, writer='pillow', fps=fps)
    plt.close()


def plot_3d_motion_global(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=20, radius=4):
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)

    ax = p3.Axes3D(fig, auto_add_to_figure=False) 
    fig.add_axes(ax)
    init()
    MINS = data.min(axis=0).min(axis=0) # [min_x, min_y, min_z]
    MAXS = data.max(axis=0).max(axis=0) # [max_x, max_y, max_z]
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1] # min_y
    # data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        global text_annotations
        # for text in text_annotations:
        #     text.remove()
        # text_annotations = []  # remove text in the current frame

        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        plot_xzPlane(
            minx=-2, 
            maxx=2, 
            miny=0, 
            minz=-2, 
            maxz=2
        ) # fix ground plane
        
        if index > 1:
            ax.plot3D(trajec[:index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1], linewidth=1.0, color='blue')
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        # show minimum y vertex
        min_y_index = np.argmin(data[index, :, 1])  
        min_y_vertex = data[index, min_y_index]    
        ax.scatter(min_y_vertex[0], min_y_vertex[1], min_y_vertex[2], color='yellow', s=50, label='Lowest Y')

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    
    ani.save(save_path, writer='pillow', fps=fps)
    plt.close()