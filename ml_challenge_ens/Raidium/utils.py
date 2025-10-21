import numpy as np 
import matplotlib.pyplot as plt 
def plot_scan(X_train, y_train, chosen_scan_to_plot=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    image = X_train[chosen_scan_to_plot]
    seg = y_train[chosen_scan_to_plot]

    # Affichage image d'origine
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Masquage des 0 (background)
    seg_masked = np.ma.masked_where(seg == 0, seg)

    # Nouvelle façon d'appeler la colormap
    cmap = plt.colormaps['tab10']  # Matplotlib ≥ 3.7
    ax2.imshow(image, cmap='gray')
    ax2.imshow(seg_masked, cmap=cmap, alpha=0.8)
    ax2.set_title('Mask Segmentation')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

