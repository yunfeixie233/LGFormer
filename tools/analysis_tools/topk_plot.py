import matplotlib.pyplot as plt

def plot_combined_top_k_iou(data, res_path=None):
    # Set up the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=500)
    
    for i, (top_k_iou_dict_part, ref_part, top_k_iou_dict_obj, ref_obj) in enumerate(data):
        ax = axs[i]
        
        # Sort and prepare part data
        sorted_keys_part = sorted(top_k_iou_dict_part.keys())
        top_k_values_part = sorted_keys_part
        iou_values_part = [top_k_iou_dict_part[k] for k in sorted_keys_part]
        ref_values_part = [ref_part[k] for k in sorted_keys_part]

        # Sort and prepare obj data
        sorted_keys_obj = sorted(top_k_iou_dict_obj.keys())
        top_k_values_obj = sorted_keys_obj
        iou_values_obj = [top_k_iou_dict_obj[k] for k in sorted_keys_obj]
        ref_values_obj = [ref_obj[k] for k in sorted_keys_obj]

        # Define colors
        light_blue = (0.3, 0.5, 1.0)
        light_red = (1.0, 0.4, 0.4)
        linewidth = 2
        markersize= 8

        # Plot part data
        ax.plot(top_k_values_part, iou_values_part, color=light_blue, marker='o', linestyle='-', linewidth=linewidth, markersize=markersize, label="IOU (Part)" if i == 0 else "")
        ax.plot(top_k_values_part, ref_values_part, color=light_blue, marker='', linestyle='--', linewidth=linewidth, markersize=markersize,label="Reference (Part)" if i == 0 else "")

        # Plot obj data
        ax.plot(top_k_values_obj, iou_values_obj, color=light_red, marker='^', linestyle='-', linewidth=linewidth, markersize=markersize,  label="IOU (Object)" if i == 0 else "")
        ax.plot(top_k_values_obj, ref_values_obj, color=light_red, marker='', linestyle='--', linewidth=linewidth, markersize=markersize, label="Reference (Object)" if i == 0 else "")

        # Set the labels and limits
        ax.set_xlabel('Numbers of Superpixels/Group Tokens', fontweight='bold', fontsize=13)
        ax.set_ylabel('mIOU', fontweight='bold', fontsize=13)
        ax.set_xlim(left=0, right=max(sorted_keys_part + sorted_keys_obj) + 1)
        ax.set_ylim(bottom=0, top=1.0)
        ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)

    # Remove individual legends from upper right corners
    for ax in axs:
        ax.legend().set_visible(False)

    # Set a common legend at the bottom center
    fig.legend(handles=axs[0].lines + axs[1].lines, labels=["IOU (Part)", "Reference (Part)", "IOU (Object)", "Reference (Object)"], loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, prop={'weight': 'bold', 'size': 11})
    
    fig.tight_layout(rect=[0, 0.1, 1, 0.9])

    # Save the plot if a path is provided
    if res_path:
        plt.savefig(res_path)

    # Show the plot
    plt.show()

# Your data
top_k_iou_dict_part = {5: 0.654, 6: 0.692, 4: 0.631, 3: 0.604, 1: 0.573, 2: 0.573}
ref_part = {5: 0.694, 4: 0.694, 3: 0.694, 2: 0.694, 1: 0.694, 6: 0.694}
top_k_iou_dict_obj = {10: 0.807, 5: 0.616, 6: 0.668, 4: 0.554, 3: 0.480, 1: 0.275, 2: 0.275, 8: 0.748, 7: 0.711, 9: 0.780}
ref_obj = {10: 0.801, 9: 0.801, 8: 0.801, 7: 0.801, 6: 0.801, 5: 0.801, 4: 0.801, 3: 0.801, 2: 0.801, 1: 0.801}

# Data for two plots
data1 = (top_k_iou_dict_part, ref_part, top_k_iou_dict_obj, ref_obj)
top_k_iou_dict_part = {1: 0.587, 2: 0.590, 3: 0.610, 4: 0.672, 5: 0.674}
ref_part = {1: 0.674, 2: 0.674, 3: 0.674, 4: 0.674, 5: 0.674}
top_k_iou_dict_obj ={1: 0.289, 2: 0.291, 3: 0.308, 4: 0.333, 5: 0.482, 6: 0.554, 7: 0.594, 8: 0.659, 9: 0.762, 10: 0.796}
ref_obj = {1: 0.798, 2: 0.798, 3: 0.798, 4: 0.798, 5: 0.798, 6: 0.798, 7: 0.798, 8: 0.798, 9: 0.798, 10: 0.798}

data2 =(top_k_iou_dict_part, ref_part, top_k_iou_dict_obj, ref_obj)

# Call the plotting function
plot_combined_top_k_iou([data1, data2], res_path='combined_plot.png')
