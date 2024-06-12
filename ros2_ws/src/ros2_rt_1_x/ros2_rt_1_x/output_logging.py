import matplotlib.pyplot as plt
import time
import numpy as np

def create_full_log(pose_history, actions, natural_language_instruction, inference_interval, filename_base):
    save_csv(pose_history, filename_base)
    draw_pose_history(pose_history, natural_language_instruction, inference_interval, filename_base)
    draw_heatmap(pose_history, filename_base)
    draw_model_output(actions, filename_base)

def save_csv(pose_history, filename_base):
    with open(f'./data/plots/{filename_base}.csv', 'w') as f:
            f.write('X,Y,Z,Roll,Pitch,Yaw,Gripper\n')
            for a in pose_history:
                f.write(f'{a[0]},{a[1]},{a[2]},{a[3]},{a[4]},{a[5]},{a[6]}\n')

def draw_pose_history(pose_history, natural_language_instruction, inference_interval, filename_base):

    fig, axs = plt.subplots(3, 3)

    axs[0,0].plot([a[0] for a in pose_history], color='green')
    axs[0,0].set_title('X')
    axs[0,0].set_ylim([-0.6, 0.6])
    axs[0,1].plot([a[1] for a in pose_history], color='green')
    axs[0,1].set_title('Y')
    axs[0,1].set_ylim([0.3, 0.8])
    axs[0,2].plot([a[2] for a in pose_history], color='green')
    axs[0,2].set_title('Z')
    axs[0,2].set_ylim([0.1, 0.7])

    axs[1,0].plot([a[3] for a in pose_history], color='green')
    axs[1,0].set_title('Roll')
    axs[1,0].set_ylim([0, 90])
    axs[1,1].plot([a[4] for a in pose_history], color='green')
    axs[1,1].set_title('Pitch')
    axs[1,1].set_ylim([0, 90])
    axs[1,2].plot([a[5] for a in pose_history], color='green')
    axs[1,2].set_title('Yaw')
    axs[1,2].set_ylim([-20, 200])

    axs[2,0].plot([a[6] for a in pose_history], color='green')
    axs[2,0].set_title('Gripper')
    axs[2,0].set_ylim([0.02, 0.08])
    axs[2,1].plot([a[7][0] for a in pose_history], color='green')
    axs[2,1].set_title('Terminate')
    axs[2,1].set_ylim([0, 1])

    axs[2,2].imshow(plt.imread('./data/tmp_inference.png'))
    axs[2,2].axis('off')

    # print date and time of inference
    # fig.suptitle(f'\"{self.natural_language_instruction}", Frequency = {round(1/self.inference_interval,1)}s')
    # second line
    fig.text(0.5, 0.97, f'"{natural_language_instruction}", Frequency = {round(1/inference_interval,1)}Hz', ha='center')
    fig.text(0.5, 0.01, f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', ha='center')

    filename = f'{filename_base}_pose_history'

    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.savefig(f'./data/plots/{filename}.png', dpi=300)

def draw_heatmap(pose_history, filename_base):

    x = [a[0] for a in pose_history]
    y = [a[1] for a in pose_history]

    plt.figure()

    plt.hist2d(x,y, bins=[np.arange(-0.6,0.6,0.06),np.arange(0.3,0.8,0.025)], cmap='Greens')
    plt.colorbar(label='Amount of coordinates in bin')

    plt.gca().invert_yaxis()

    plt.scatter(0,0.3, color='grey', marker=6, s=100)
    plt.scatter(0,0.325, color='grey', marker="$Robot Base$", s=900, linewidths=0.2)
    
    plt.title('Heatmap of 2D Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.xticks([-0.6,-0.3,0,0.3,0.6])
    plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8])

    filename = f'{filename_base}_heatmap'

    plt.savefig(f'./data/plots/{filename}.png', dpi=300)

def draw_model_output(actions, filename_base):
    data = actions

    fig, axs = plt.subplots(1, 10)

    fig.set_size_inches([45, 5])
    fig.subplots_adjust(left=0.03, right=0.97)

    axs[0].plot([a['terminate_episode'][0] for a in data], color='red')
    axs[0].set_title('terminate_episode_0')
    axs[1].plot([a['terminate_episode'][1] for a in data], color='red')
    axs[1].set_title('terminate_episode_1')
    axs[2].plot([a['terminate_episode'][2] for a in data], color='red')
    axs[2].set_title('terminate_episode_2')
    axs[3].plot([a['world_vector'][0] for a in data], color='red')
    axs[3].set_title('world_vector_0')
    axs[4].plot([a['world_vector'][1] for a in data], color='red')
    axs[4].set_title('world_vector_1')
    axs[5].plot([a['world_vector'][2] for a in data], color='red')
    axs[5].set_title('world_vector_2')
    axs[6].plot([a['rotation_delta'][0] for a in data], color='red')
    axs[6].set_title('rotation_delta_0')
    axs[7].plot([a['rotation_delta'][1] for a in data], color='red')
    axs[7].set_title('rotation_delta_1')
    axs[8].plot([a['rotation_delta'][2] for a in data], color='red')
    axs[8].set_title('rotation_delta_2')
    axs[9].plot([a['gripper_closedness_action'][0] for a in data], color='red')
    axs[9].set_title('gripper_closedness_action_0')

    filename = f'{filename_base}_model_output'

    fig.savefig(f'./data/plots/{filename}.png')


