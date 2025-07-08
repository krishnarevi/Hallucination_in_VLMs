import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"D:\Uni\LangNVision\Project\Hallucination_in_VLMs\output\results_table.csv")

colors = [ 'green', 'blue', 'orange'] 
plt.figure(figsize=(8, 6)) 

for i, row in df.iterrows():
    plt.scatter(row['CLIP Score'], row['DreamSim Score'],
                s=100,            # Marker size
                color=colors[i],  # Assign unique color from the 'colors' list
                label=row['Model'] # Label for the legend
               )

for i, row in df.iterrows():
    plt.annotate(row['Model'],
                 (row['CLIP Score'] + 0.005, row['DreamSim Score'] + 0.01), # Position of the text
                 fontsize=10,
                 ha='left',      # Horizontal alignment
                 va='bottom'     # Vertical alignment
                )

plt.xlabel('CLIP Score', fontsize=14)
plt.ylabel('DreamSim', fontsize=14)
plt.xlim(0, 0.4)
plt.ylim(0, 0.7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks([i * 0.05 for i in range(9)]) # Ticks at 0.0, 0.05, 0.10, ..., 0.40
plt.yticks([i * 0.1 for i in range(8)])  # Ticks at 0.0, 0.1, 0.2, ..., 0.7
plt.legend(loc='upper right') 
plt.tight_layout()
plt.savefig('model_performance_scatter_plot_colored.png')
plt.show()