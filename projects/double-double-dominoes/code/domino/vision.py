from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import typing as t
from typing import Tuple, List
import numpy as np
from numpy import ndarray
import cv2 as cv
import abc


T_OUT = t.TypeVar('T_OUT')
T_IN = t.TypeVar('T_IN')


class Operation(abc.ABC, t.Generic[T_IN, T_OUT]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def __call__(self, image: T_IN) -> T_OUT:
        raise NotImplementedError()


class DisplayOperation(Operation[np.ndarray, None]):
    def __init__(self, *args, scale=1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = scale

    def __call__(self, image: np.ndarray) -> None:
        # Scale down images to speedup computations
        image = cv.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv.INTER_LINEAR)

        # Display the image and wait for input
        cv.imshow('image', image)
        cv.waitKey(0)
        cv.destroyAllWindows()


class PreProcessOperation(Operation[np.ndarray, np.ndarray]):
    def __init__(self, *args, scale: float=1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale: float = scale

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Scale down images to speedup computations
        image = cv.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv.INTER_LINEAR)

        # From uint8 to float32
        image = image.astype(np.float32) / 255.

        # Stop
        return image


class PerspectiveOperation(Operation[t.Any, np.ndarray]):
    def __init__(self, *args, show_image: bool=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Internal matching algorithm for finding good keypoint pairs
        self.__matcher_sift: FeatureMatching = SIFTFeatureMatching(show_image=show_image, min_lowe_ratio=0.55, max_lowe_ratio=0.8, min_matches=10)
        self.__matcher_orb: FeatureMatching = ORBFeatureMatching(show_image=show_image)
        self.matcher: FeatureMatching = self.__matcher_sift
        self.show_image: bool = show_image

    def __call__(self, target: np.ndarray, template: np.ndarray) -> np.ndarray:
        # Compute the location of relevant keypoints in both images using SIFT or ORB
        try:
            self.matcher = self.__matcher_sift
            target_pts, template_pts = self.matcher(target, template)
        except RuntimeError:
            self.matcher = self.__matcher_orb
            target_pts, template_pts = self.matcher(target, template)

        # Find the transformation from the target_pts to the template_pts
        M, mask = cv.findHomography(target_pts, template_pts, cv.RANSAC, 5.0, maxIters=3_000)
        matchesMask = mask.ravel().tolist()

        # Display the good matches
        if self.show_image is True:
            params_draw = dict(matchesMask=matchesMask, matchColor=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            knn_image: np.ndarray = cv.drawMatches(target, self.matcher.target_kpts_, template, self.matcher.template_kpts_, self.matcher.relevant_matches_, None, **params_draw)
            plt.figure(figsize=(15, 15))
            plt.imshow(knn_image)
            plt.show()

        # Apply the H transformation over the input image
        warped_target = cv.warpPerspective(target, M, template.shape, None)

        # Display the result
        if self.show_image:
            cv.imshow('Warped Target using H', warped_target)
            cv.waitKey(0)
            cv.destroyWindow('Warped Target using H')
        return warped_target


class Image2GridOperation(Operation[np.ndarray, np.ndarray]):
    def __init__(self, *args, show_image: bool=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.show_image: bool = show_image

    def __call__(self, image: ndarray) -> ndarray:
        # Smooth before detecting edges to remove noise
        image_blurred: np.ndarray = cv.GaussianBlur(image, (7, 7), 0)
        # self.__show_blur(image_blurred)

        # Use Canny's Edge Detector to extract a binary image of edges
        image_edges: np.ndarray = cv.Canny(image_blurred, threshold1=50, threshold2=100, apertureSize=3, L2gradient=True)
        self.__show_edges(image, image_edges)

        # Detect the outer layer
        lines: t.List[t.List[t.List[float]]] = cv.HoughLines(image_edges, 1, np.pi / 180, 250, None)

        # Display the initial lines that were found
        image_edges_copy: np.ndarray = cv.cvtColor(image_edges.copy(), cv.COLOR_GRAY2BGR)
        horizontal_lines: t.List[t.List[float]] = []
        vertical_lines: t.List[t.List[float]] = []
        good_lines: t.List[t.List[float]] = []

        # 1st stage filtering - Keep only horizontal & vertical lines
        for line in lines:
            rho, theta = line[0]
            angle: float = theta * 180 / np.pi

            # From Polar to Cartesian
            a:  float = np.cos(theta)
            b:  float = np.sin(theta)
            x0: float = a * rho
            y0: float = b * rho
            x1 = int(x0 + 1000 * -b)
            y1 = int(y0 + 1000 *  a)
            x2 = int(x0 - 1000 * -b)
            y2 = int(y0 - 1000 *  a)

            # Keep only vertical and horizontal lines
            if (angle > 170 and angle < 190) or (angle < 10 or angle > 350):
                horizontal_lines.append([x1, y1, x2, y2])
            elif (angle > 80 and angle < 100) or (angle > 260 and angle < 280):
                vertical_lines.append([x1, y1, x2, y2])

        good_lines = [*vertical_lines, *horizontal_lines]
        for line in good_lines:
            x1, y1, x2, y2 = line
            cv.line(image_edges_copy, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow('Lines Found from Edges', image_edges_copy)
        cv.waitKey(0)
        cv.destroyWindow('Lines Found from Edges')
        return image_edges

    # def __call__(self, image: ndarray) -> ndarray:
    #     # Smooth before detecting edges to remove noise
    #     image_blurred: np.ndarray = cv.GaussianBlur(image, (7, 7), 0)
    #     # self.__show_blur(image_blurred)

    #     # Use Canny's Edge Detector to extract a binary image of edges
    #     image_edges: np.ndarray = cv.Canny(image_blurred, threshold1=50, threshold2=100, apertureSize=3, L2gradient=True)
    #     self.__show_edges(image, image_edges)

    #     # Detect the outer layer
    #     lines: t.List[t.List[t.List[float]]] = cv.HoughLines(image_edges, 1, np.pi / 180, 250, None)

    #     # Display the initial lines that were found
    #     image_edges_copy: np.ndarray = cv.cvtColor(image_edges.copy(), cv.COLOR_GRAY2BGR)
    #     horizontal_lines: t.List[t.List[float]] = []
    #     vertical_lines: t.List[t.List[float]] = []
    #     good_lines: t.List[t.List[float]] = []

    #     # 1st stage filtering - Keep only horizontal & vertical lines
    #     for line in lines:
    #         rho, theta = line[0]
    #         angle: float = theta * 180 / np.pi

    #         # From Polar to Cartesian
    #         a:  float = np.cos(theta)
    #         b:  float = np.sin(theta)
    #         x0: float = a * rho
    #         y0: float = b * rho
    #         x1 = int(x0 + 1000 * -b)
    #         y1 = int(y0 + 1000 *  a)
    #         x2 = int(x0 - 1000 * -b)
    #         y2 = int(y0 - 1000 *  a)

    #         # Keep only vertical and horizontal lines
    #         if (angle > 170 and angle < 190) or (angle < 10 or angle > 350):
    #             horizontal_lines.append([x1, y1, x2, y2])
    #         elif (angle > 80 and angle < 100) or (angle > 260 and angle < 280):
    #             vertical_lines.append([x1, y1, x2, y2])

    #     good_lines = [*vertical_lines, *horizontal_lines]
    #     for line in good_lines:
    #         x1, y1, x2, y2 = line
    #         cv.line(image_edges_copy, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)

    #     cv.imshow('Lines Found from Edges', image_edges_copy)
    #     cv.waitKey(0)
    #     cv.destroyWindow('Lines Found from Edges')
    #     return image_edges

    def __show_blur(self, image_blurred: np.ndarray) -> None:
        if self.show_image:
            cv.imshow('Blurred Image', image_blurred)
            cv.waitKey(0)
            cv.destroyWindow('Blurred Image')

    def __show_edges(self, image: np.ndarray, edges: np.ndarray) -> None:
        if self.show_image:
            _, ax = plt.subplots(1, 2, figsize=(15, 15))
            ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            ax[0].set_title('Original Image')
            ax[1].imshow(edges, cmap='gray', vmin=0, vmax=255)
            ax[1].set_title('Edges in the Image')


class FeatureMatching(abc.ABC):
    def __init__(self, *args, show_image: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._show_image: bool = show_image

        # Computed after a call to _extract_features
        self.target_kpts_: t.Any
        self.target_dpts_: t.Any
        self.template_kpts_: t.Any
        self.template_dpts_: t.Any
        self.relevant_matches_: t.List[cv.DMatch]

    @abc.abstractmethod
    def __call__(self, target: np.ndarray, template: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _extract_features(self, image: np.ndarray) -> t.Tuple[t.Any, t.Any]:
        raise NotImplementedError()

    def _show_image_keypoints(self, image: np.ndarray, keypoints: t.Any) -> None:
        if self._show_image is True:
            kpts_image = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imshow('Image Keypoints', kpts_image)
            cv.waitKey(0)
            cv.destroyWindow('Image Keypoints')


class SIFTFeatureMatching(FeatureMatching):
    def __init__(self, *args, min_lowe_ratio: float = 0.55, max_lowe_ratio: float = 0.8, min_matches: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sift: t.Any = cv.SIFT_create()
        self.min_match_count: int = min_matches
        self.min_lowe_ratio: float = min_lowe_ratio
        self.max_lowe_ratio: float = max_lowe_ratio

    def __call__(self, target: ndarray, template: ndarray) -> Tuple[ndarray, ndarray]:
        # Extract features using SIFT
        self.target_kpts_, self.target_dpts_ = self._extract_features(target)
        self.template_kpts_, self.template_dpts_ = self._extract_features(template)

        # Perform matching
        matcher: cv.BFMatcher = cv.BFMatcher(normType=cv.NORM_L2)
        matches: t.List[t.List[cv.DMatch]] = matcher.knnMatch(self.target_dpts_, self.template_dpts_, k=2)

        # Filter using Lowe's Ratio Test
        lowe_ratio: float = self.min_lowe_ratio
        self.relevant_matches_: t.List[cv.DMatch] = []

        # Retry filtering until an upper_bound on the ratio
        while len(self.relevant_matches_) < self.min_match_count and lowe_ratio < self.max_lowe_ratio:
            self.relevant_matches_ = []
            for first, second in matches:
                if first.distance < lowe_ratio * second.distance:
                    self.relevant_matches_.append(first)
            lowe_ratio = np.clip(lowe_ratio + 0.1, 0.0, self.max_lowe_ratio)

        # Ensure that at least a solution was found
        if lowe_ratio == self.max_lowe_ratio:
            raise RuntimeError('Not enough good matches were found using SIFT')

        # Fetch the locations of the keypoints in both images
        src_pts: np.ndarray = np.float32([ self.target_kpts_[m.queryIdx].pt for m in self.relevant_matches_ ]).reshape(-1,1,2)
        dst_pts: np.ndarray = np.float32([ self.template_kpts_[m.trainIdx].pt for m in self.relevant_matches_ ]).reshape(-1,1,2)
        return src_pts, dst_pts

    def _extract_features(self, image: ndarray) -> t.Tuple[t.Any, t.Any]:
        kpts, dpts = self.sift.detectAndCompute(image, None)
        self._show_image_keypoints(image, kpts)
        return kpts, dpts


class ORBFeatureMatching(FeatureMatching):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.orb_ = cv.ORB_create()

    def __call__(self, target: ndarray, template: ndarray) -> Tuple[ndarray, ndarray]:
        # Extract features using SIFT
        self.target_kpts_, self.target_dpts_ = self._extract_features(target)
        self.template_kpts_, self.template_dpts_ = self._extract_features(template)

        # Perform matching
        matcher: cv.BFMatcher = cv.BFMatcher(normType=cv.NORM_HAMMING, crossCheck=True)
        self.relevant_matches_: t.List[cv.DMatch] = matcher.match(self.target_dpts_, self.template_dpts_)

        # Fetch the locations of the keypoints in both images
        src_pts: np.ndarray = np.float32([ self.target_kpts_[m.queryIdx].pt for m in self.relevant_matches_ ]).reshape(-1,1,2)
        dst_pts: np.ndarray = np.float32([ self.template_kpts_[m.trainIdx].pt for m in self.relevant_matches_ ]).reshape(-1,1,2)
        return src_pts, dst_pts

    def _extract_features(self, image: ndarray) -> t.Tuple[t.Any, t.Any]:
        kpts, dpts = self.orb_.detectAndCompute(image, None)
        self._show_image_keypoints(image, kpts)
        return kpts, dpts

