import numpy as np
import cv2

def singleScaleRetinex(img,variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex

def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex

   
def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex

def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex

def evaluate_retinex_algorithms(algorithms: dict[str, np.ndarray], day_image: np.ndarray) -> None:
    # Channel-wise MSE comparison between day and final processed night image
    for algorithm, processed_night_image in algorithms.items():
        mse_channels = np.mean((day_image.astype("float") - processed_night_image.astype("float")) ** 2, axis=(0, 1))
        mse_r, mse_g, mse_b = mse_channels
        mse_overall = (mse_r + mse_g + mse_b) / 3

        print(f"Evaluation for {algorithm}:", f"R MSE={mse_r:.2f}, G MSE={mse_g:.2f}, B MSE={mse_b:.2f}, Overall MSE={mse_overall:.2f}")

def post_process(img):
    from cdf_histogram_matching import match_histograms_cdf

    img = cv2.medianBlur(img, 2)
    img = cv2.fastNlMeansDenoisingColored(img, None, 20, 5, 7, 21)
    img = match_histograms_cdf(img, cv2.imread('data/day.jpg'))

    return img

def main():
    variance_list=[15, 80, 30]
    variance=300
        
    img = cv2.imread('data/night.jpg')
    img_msr=MSR(img,variance_list)
    img_ssr=SSR(img, variance)

    img_msr = post_process(img_msr)
    img_ssr = post_process(img_ssr)


    cv2.imshow('Original', img)
    cv2.imshow('MSR', img_msr)
    cv2.imshow('SSR', img_ssr)
    cv2.imwrite('SSR.jpg', img_ssr)
    cv2.imwrite('MSR.jpg',img_msr)

    evaluate_retinex_algorithms({'MSR': img_msr, 'SSR': img_ssr}, day_image=cv2.imread('data/day.jpg'))


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":    main()