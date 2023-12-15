import matplotlib.pyplot as plt
from matplotlib import font_manager

# This will print a list of available fonts
available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
print(available_fonts)





def plot_top_k_vs_object_iou(top_k_iou_dict, ref = None,res_path=None):
    # Extracting top_k and object_iou values
    sorted_keys = sorted(top_k_iou_dict.keys())
    top_k_values = sorted_keys
    object_iou_values = [top_k_iou_dict[k] for k in sorted_keys]
        

    plt.clf()
    font_kwparams = {'fontname': 'Times New Roman', 'fontsize': 13} 
    legend_kwparams = {'family': 'Times New Roman', 'size': 13}    
       
    plt.plot(top_k_values, object_iou_values, 'b-', label='Top-K group tokens to form object segment',)  # 使用蓝色实线
    if ref is not None:
        sorted_keys = sorted(ref.keys())
        top_k_values = sorted_keys
        object_iou_values = [ref[k] for k in sorted_keys]        
        plt.plot(top_k_values, object_iou_values, 'r--',label='group tokens under supervision for object IOU',)  # 使用蓝色实线
        
    # Setting font parameters


    # Setting the limits and labels for the plot
    plt.xlim(left=0, right=max(top_k_values) + 1)  # X轴从0开始
    plt.ylim(bottom=0.0, top=1.0)  # Y轴从0开始
    # plt.ylim(bottom=0.5, top=0.8)  # Y轴从0开始
    plt.xlabel('Top K Group Tokens', **font_kwparams)
    plt.ylabel('Object IOU', **font_kwparams)
    # Adding legend
    plt.legend(prop=legend_kwparams)

    # Saving the plot if a path is provided
    if res_path:
        plt.savefig(res_path)

    plt.show()

# Example Usage
# top_k_iou_dict_part = {5: 0.654,6:0.692,4:0.631,3:0.604 ,1:0.573 ,2:0.573,}
# ref_part = {5: 0.694,4: 0.694,3: 0.694,2: 0.694,1: 0.694,6: 0.694}
ref_obj = {10:0.801,9:0.801,8:0.801,7:0.801,6:0.801,5:0.801,4: 0.801,3: 0.801,2: 0.801,1: 0.801,6: 0.801}

top_k_iou_dict_obj = {10:0.807,5:0.616,6:0.668,4:0.554,3:0.480,1:0.275,2:0.275,8:0.748,7:0.711,9:0.780}

plot_top_k_vs_object_iou(top_k_iou_dict_obj,ref_obj,res_path = '/data1/yunfei/obj_from_part.png')
