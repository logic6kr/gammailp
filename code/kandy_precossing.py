#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

def transfer(index = '00', label = 'true', task_name = 'twopairs-with-same-and-different-colors', data_set = 'train'):
    # Load image
    if len(index) == 2:
        index = '0' + index
    image = cv2.imread(f'gammaILP/dat-kandinsky-patterns/{task_name}/{label}/000{index}.png')
    save_folde  = f'gammaILP/dat-kandinsky-patterns/{task_name}/{data_set}/{label}/cropped_objects/{index}/'
    if not os.path.exists(save_folde):
        os.makedirs(save_folde)
    # Convert image to float for comparison
    image_float = image.astype(np.float32)

    # Calculate channel difference: grey pixels have R ≈ G ≈ B
    max_rgb = np.max(image_float, axis=2)
    min_rgb = np.min(image_float, axis=2)
    diff = max_rgb - min_rgb

    # Threshold: mark pixels that are "non-grey"
    threshold = 10
    non_grey_mask = (diff > threshold).astype(np.uint8) * 255

    # Clean mask using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_grey_mask = cv2.morphologyEx(non_grey_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of non-grey regions
    contours, _ = cv2.findContours(non_grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop and save each non-grey object
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = image[y:y+h, x:x+w]
        cv2.imwrite(f"{save_folde}/object_{i}.png", cropped)

    # Optional: visualize using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Non-Grey Mask")
    plt.imshow(non_grey_mask, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_folde}/result_preview.png")  # Saves the preview
    plt.show()

def main(task_name= 'onetriangle', training_numbers=15, testing_numbers = 10):
    # Example usage
    # task_name = 'twopairs-with-same-and-different-colors'
    print(f"Processing task: {task_name}") 
    for index in range(0, training_numbers):
        if index < 10:
            index = '0' + str(index)
        else:
            index = str(index)
        print(f"Processing index: {index} for train set")
        transfer(index=index, label='true', task_name =task_name, data_set='train')
        transfer(index=index, label='false', task_name =task_name, data_set='train')
    

    for index in range(training_numbers, training_numbers + testing_numbers):
        if index < 10:
            index = '0' + str(index)
        else:
            index = str(index)
        print(f"Processing index: {index} for test set")
        transfer(index=index, label='true', task_name =task_name, data_set='test')
        transfer(index=index, label='false', task_name =task_name, data_set='test')

# %%
if __name__ == "__main__":
    # Example usage
    # task_name = 'twopairs-with-same-and-different-colors'
    main(task_name='onered_all', max_index=15)