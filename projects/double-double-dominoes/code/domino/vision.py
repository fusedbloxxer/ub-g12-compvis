from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import typing as t
from typing import Tuple, List
import numpy as np
from numpy import ndarray
import skimage as ski
import torchvision as tv
import torch
import cv2 as cv
import abc

from .model import CellClassifier
from .util import same_cell_dim


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
        return cv.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv.INTER_LINEAR)


class PerspectiveOperation(Operation[np.ndarray, np.ndarray]):
    def __init__(self, board_template: np.ndarray, *args, show_image: bool=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Internal matching algorithm for finding good keypoint pairs
        self.__matcher_sift: FeatureMatching = SIFTFeatureMatching(show_image=show_image, min_lowe_ratio=0.55, max_lowe_ratio=0.8, min_matches=10)
        self.__matcher_orb: FeatureMatching = ORBFeatureMatching(show_image=show_image)
        self.matcher: FeatureMatching = self.__matcher_sift
        self.template: np.ndarray = board_template
        self.show_image: bool = show_image

    def __call__(self, target: np.ndarray) -> np.ndarray:
        # Compute the location of relevant keypoints in both images using SIFT or ORB
        try:
            self.matcher = self.__matcher_sift
            target_pts, template_pts = self.matcher(target, self.template)
        except RuntimeError:
            self.matcher = self.__matcher_orb
            target_pts, template_pts = self.matcher(target, self.template)

        # Find the transformation from the target_pts to the template_pts
        M, mask = cv.findHomography(target_pts, template_pts, cv.RANSAC, 5.0, maxIters=3_000)
        matchesMask = mask.ravel().tolist()

        # Display the good matches
        if self.show_image is True:
            params_draw = dict(matchesMask=matchesMask, matchColor=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            knn_image: np.ndarray = cv.drawMatches(target, self.matcher.target_kpts_, self.template, self.matcher.template_kpts_, self.matcher.relevant_matches_, None, **params_draw)
            plt.figure(figsize=(15, 15))
            plt.imshow(knn_image)
            plt.show()

        # Apply the H transformation over the input image
        warped_target = cv.warpPerspective(target, M, self.template.shape, None)

        # Display the result
        if self.show_image:
            cv.imshow('Warped Target using H', warped_target)
            cv.waitKey(0)
            cv.destroyWindow('Warped Target using H')
        return warped_target


class Image2GridOperation(Operation[np.ndarray, t.Tuple[np.ndarray, np.ndarray, t.List[t.List[np.ndarray]]]]):
    def __init__(self, grid_template: np.ndarray, *args, cell_splitter: str = 'hough', show_image: bool=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Create factories for later usage
        self.window_cell_extractor_factory: t.Callable[[], CellExtractor] = lambda: WindowCellExtractor(**kwargs)
        self.hough_cell_extractor_factory: t.Callable[[], CellExtractor] = lambda: HoughCellExtractor(show_image=show_image, **kwargs)

        # Determine the method used to fetch the grid tiles
        if cell_splitter == 'window':
            self.cell_extractor_factory: t.Callable[[], CellExtractor] = self.window_cell_extractor_factory
        elif cell_splitter == 'hough':
            self.cell_extractor_factory: t.Callable[[], CellExtractor] = self.hough_cell_extractor_factory
        else:
            raise NotImplementedError()

        # Other params
        self.grid_template: np.ndarray = grid_template
        self.show_image: bool = show_image

    def __call__(self, image: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray, t.List[t.List[np.ndarray]]]:
        # Unpack the argument
        target_image, template_image = image, self.grid_template

        # Smooth before detecting edges to remove noise
        template_image_blurred: np.ndarray = cv.GaussianBlur(template_image.copy(), (5, 5), 0)
        target_image_blurred: np.ndarray = cv.GaussianBlur(target_image.copy(), (5, 5), 0)
        self.__show_blur(template_image_blurred)
        self.__show_blur(target_image_blurred)

        # Use Canny's Edge Detector to extract a binary image of edges
        template_image_edges: np.ndarray = cv.Canny(template_image_blurred, threshold1=35, threshold2=90, apertureSize=3, L2gradient=True)
        target_image_edges: np.ndarray = cv.Canny(target_image_blurred, threshold1=35, threshold2=90, apertureSize=3, L2gradient=True)
        self.__show_edges(template_image, template_image_edges)
        self.__show_edges(target_image, target_image_edges)

        # Template Matching
        match: np.ndarray = cv.matchTemplate(target_image_edges, template_image_edges, cv.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)

        # Extract the grid borders
        top_left: t.Tuple[int, int] = min_loc
        bot_right: t.Tuple[int, int] = (top_left[0] + template_image.shape[1], top_left[1] + template_image.shape[0])
        grid_region: np.ndarray = target_image[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]
        self.__show_match(target_image, grid_region, top_left, bot_right)

        # Extract the grid cells using the given method otherwise fallback to window extractor
        try:
            cell_extractor: CellExtractor = self.cell_extractor_factory()
            cells: t.List[t.List[np.ndarray]] = cell_extractor(grid_region)
        except RuntimeError:
            if self.show_image:
                print('Fallback to WindowCellExtractor')
            cell_extractor: CellExtractor = self.window_cell_extractor_factory()
            cells: t.List[t.List[np.ndarray]] = cell_extractor(grid_region)
        grid_patch: np.ndarray = self.__cells_to_image(cells)
        self.__show_grid(grid_region, grid_patch)
        return grid_region, grid_patch, cells

    def __cells_to_image(self, cells: t.List[t.List[np.ndarray]]) -> np.ndarray:
        nrows, ncols = len(cells), len(cells[0])
        sizes: t.List[t.Tuple[int, int]] = [cell.shape for line in cells for cell in line]
        cell_height: int = max(map(lambda x: x[0], sizes))
        cell_width:  int = max(map(lambda x: x[1], sizes))

        gap: int = 10
        grid_size_y: int = nrows * cell_height + nrows * gap
        grid_size_x: int = ncols *  cell_width + ncols * gap
        grid_size: t.Tuple[int, int] = (grid_size_y, grid_size_x)
        grid_patch: np.ndarray = np.zeros(grid_size, dtype=np.uint8)

        for i in range(nrows):
            for j in range(ncols):
                grid_y = slice(i * (cell_height + gap), i * (cell_height + gap) + cells[i][j].shape[0])
                grid_x = slice(j * ( cell_width + gap), j * ( cell_width + gap) + cells[i][j].shape[1])
                grid_patch[grid_y, grid_x] = cells[i][j]
        return grid_patch

    def __show_grid(self, image: np.ndarray, grid_patch: np.ndarray) -> None:
        if self.show_image:
            fig: Figure = plt.figure(figsize=(20, 20))
            top_level_grid = GridSpec(nrows=1, ncols=2, figure=fig)

            ax_grid = fig.add_subplot(top_level_grid[0])
            ax_grid.set_title('Initial Grid')
            ax_grid.imshow(image, cmap='gray')

            ax_grid_patch = fig.add_subplot(top_level_grid[1])
            ax_grid_patch.imshow(grid_patch, cmap='gray', vmin=0, vmax=255)
            ax_grid_patch.set_title('Grid Made of Patches')
            plt.show()

    def __show_match(self, image: np.ndarray, grid: np.ndarray, top_left: t.Tuple[int, int], bot_right: t.Tuple[int, int]) -> None:
        if self.show_image:
            _, ax = plt.subplots(1, 2, figsize=(15, 15))
            image = image.copy()
            cv.rectangle(image, top_left, bot_right, 255, 2)
            ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
            ax[0].set_title('Detected Region of Interest')
            ax[1].imshow(grid, cmap='gray', vmin=0, vmax=255)
            ax[1].set_title('Grid')
            plt.show()

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


class Board2GridOperation(Operation[np.ndarray, t.Tuple[np.ndarray, np.ndarray, t.List[t.List[np.ndarray]]]]):
    def __init__(self, board_template: np.ndarray, grid_template: np.ndarray, *args, cell_splitter: str='hough', show_image: bool=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Internal ops
        self.op_perspective = PerspectiveOperation(board_template=board_template, show_image=show_image)
        self.op_img2grid    = Image2GridOperation(grid_template=grid_template, show_image=show_image, cell_splitter=cell_splitter)
        self.op_preprocess  = PreProcessOperation(scale=0.35)

    def __call__(self, image: ndarray) -> Tuple[ndarray, ndarray, List[List[ndarray]]]:
        self.image_: np.ndarray = image
        image = self.op_preprocess(image)
        image = self.op_perspective(image)
        return self.op_img2grid(image)


class Board2MatrixOpeation(Operation[np.ndarray, np.ndarray]):
    def __init__(self, board_template: np.ndarray, grid_template: np.ndarray, cell_classifier: CellClassifier, *args, cell_splitter: str='hough', show_matrix: bool=False, show_image: bool=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Internal ops
        self.op_board2grid: Board2GridOperation = Board2GridOperation(board_template=board_template, grid_template=grid_template, cell_splitter=cell_splitter, show_image=show_image)
        self.cell_classifier: CellClassifier = cell_classifier
        self.show_matrix: bool = show_matrix

    def __call__(self, image: ndarray) -> ndarray:
        self.grid_region_, self.grid_patch_, self.grid_cells_ = self.op_board2grid(image)
        self.grid_matrix_: np.ndarray = self.cell_classifier(self.grid_cells_)
        self.__show_matrix_prediction(self.grid_region_, self.grid_cells_, self.grid_matrix_)
        return self.grid_matrix_

    def __show_matrix_prediction(self, grid_image: np.ndarray, grid_cells: t.List[t.List[np.ndarray]], grid_matrix: np.ndarray) -> None:
        if not self.show_matrix:
            return
        # Ensure all grid cells are of the same size
        aligned_grid_cells = same_cell_dim(grid_cells, dsize=(45, 45)).reshape((*grid_matrix.shape, 45, 45))

        # Present the initial grid along the predicted cell contents
        f, ax = plt.subplots(1, 2, figsize=(15, 15))

        # Show the original grid
        ax[0].imshow(grid_image, cmap='gray', vmin=0, vmax=255)
        ax[0].set_title('Board Grid')

        # Show the predicted values
        cells: t.List[torch.Tensor] = []
        for i in range(grid_matrix.shape[0]):
            for j in range(grid_matrix.shape[1]):
                grid_cell: np.ndarray = cv.cvtColor(aligned_grid_cells[i][j].copy(), cv.COLOR_GRAY2BGR)
                grid_cell = cv.putText(grid_cell, str(grid_matrix[i][j]), (np.array(grid_cell.shape[:2]) // 2).tolist(), cv.FONT_HERSHEY_COMPLEX, 0.75, (204, 0, 0), 2)
                cells.append(torch.from_numpy(grid_cell))
        matrix_cells = torch.stack(cells, dim=-1).permute((3, 2, 0, 1))
        matrix_image = tv.utils.make_grid(matrix_cells, nrow=grid_matrix.shape[0], padding=3)
        ax[1].imshow(matrix_image.permute((1, 2, 0)), cmap='gray', vmin=0, vmax=255)
        ax[1].set_title('Matrix')
        plt.show()


class CellExtractor(abc.ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def __call__(self, grid: np.ndarray) -> t.List[t.List[np.ndarray]]:
        raise NotImplementedError()

    @staticmethod
    def factory(method: str, *args, **kwargs) -> 'CellExtractor':
        if method == 'window':
            return WindowCellExtractor(*args, **kwargs)
        elif method == 'hough':
            return HoughCellExtractor(*args, **kwargs)
        else:
            raise NotImplementedError()


class HoughCellExtractor(CellExtractor):
    def __init__(self,
                 *args,
                 show_image: bool=False,
                 show_eliminations: bool=False,
                 grid_gap_threshold: t.Tuple[float, float]=(39, 39),
                 inner_gap: t.Tuple[int, int] = (1, 1),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.show_image: bool = show_image
        self.show_eliminations: bool = show_eliminations
        self.grid_gap_threshold: t.Tuple[float, float] = grid_gap_threshold
        self.inner_gap: t.Tuple[int, int] = inner_gap

    def __call__(self, grid: ndarray) -> t.List[t.List[np.ndarray]]:
        try:
            # Filter out noise, apply thresholding and extract edges using Canny Algorithm
            grid_edges: np.ndarray = self.__extract_edges(grid)

            # Extract grid lines using Hough Transform
            horizontal_lines, vertical_lines = self.__extract_lines(grid_edges)

            # Ensure that a fixed amount of grid lines were found
            if len(horizontal_lines) != 16 or len(vertical_lines) != 16:
                raise RuntimeError('Wrong amount of grid lines were found using Hough Transform: {}, {}'.format(len(horizontal_lines), len(vertical_lines)))

            # Extract the intersections where the cell contents lie
            patches: t.List[t.List[np.ndarray]] = []
            for i, row_idx in enumerate(range(len(horizontal_lines) - 1)):
                patches.append([])
                for j, col_idx in enumerate(range(len(vertical_lines) - 1)):
                    curr_row: t.List[float] = horizontal_lines[row_idx]
                    next_row: t.List[float] = horizontal_lines[row_idx + 1]
                    y_min: float = curr_row[1] + self.inner_gap[1]
                    y_max: float = next_row[1] - self.inner_gap[1]

                    curr_col: t.List[float] = vertical_lines[col_idx]
                    next_col: t.List[float] = vertical_lines[col_idx + 1]
                    x_min: float = curr_col[0] + self.inner_gap[1]
                    x_max: float = next_col[0] - self.inner_gap[1]

                    patch: np.ndarray = grid[y_min: y_max, x_min: x_max]
                    patches[-1].append(patch)
            return patches
        except Exception as e:
            raise RuntimeError('Could not extract lines using Hough: {}'.format(e))

    def __extract_lines(self, grid_edges: np.ndarray) -> t.Tuple[t.List[t.List[float]], t.List[t.List[float]]]:
        # Obtain hough lines
        lines: t.List[t.List[t.List[float]]] = cv.HoughLinesP(cv.dilate(grid_edges, None), 1, np.pi / 180, 250, None, minLineLength=200, maxLineGap=250)
        horizontal_lines: t.List[t.List[float]] = []
        vertical_lines: t.List[t.List[float]] = []

        # 1st stage filtering - Keep only horizontal & vertical lines
        for line in lines:
            # Determine the type of line
            x1, y1, x2, y2 = line[0]
            angle: float = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            is_vertical:   bool = angle == -90
            is_horizontal: bool = angle ==   0

            # Group the lines based on their inclination
            if is_horizontal:
                horizontal_lines.append([x1, y1, x2, y2])
                continue
            if is_vertical:
                vertical_lines.append([x1, y1, x2, y2])

        # 2nd stage filtering - Eliminate close lines
        def show_lines(lines: t.Any) -> None:
            if self.show_image:
                # Display the results
                edges_copy: np.ndarray = cv.cvtColor(grid_edges.copy(), cv.COLOR_GRAY2BGR)
                for line in lines:
                    x1, y1, x2, y2 = line
                    cv.line(edges_copy, (x1, y1), (x2, y2), (0, 0, 255), 2, cv.LINE_AA)
                cv.imshow('Lines Found from Edges', edges_copy)
                cv.waitKey(0)
                cv.destroyWindow('Lines Found from Edges')
        def eliminate_close_lines(lines: t.List[t.List[float]], axis: int) -> t.List[t.List[float]]:
            # Collect similar lines in a bag and create a prototype (kmeans with 1 step)
            unique: t.List[t.List[float]] = [lines[0]]
            for i in range(1, len(lines)):
                # Check the first point
                if lines[i][axis] - unique[-1][axis] < self.grid_gap_threshold[axis]:
                    continue
                # Check the second point
                if lines[i][2 + axis] - unique[-1][2 + axis] < self.grid_gap_threshold[axis]:
                    continue
                # Check the center
                unique.append(lines[i])
                if self.show_eliminations:
                    show_lines(unique)
            return unique

        # Pass through each line in order as they come
        horizontal_lines = sorted(horizontal_lines, key=lambda line: line[1])
        vertical_lines = sorted(vertical_lines, key=lambda line: line[0])
        show_lines([*horizontal_lines, *vertical_lines])

        # Remove them based on the difference between the centers (works for both horiz & vertical)
        horizontal_lines = eliminate_close_lines(horizontal_lines, axis=1)
        vertical_lines = eliminate_close_lines(vertical_lines, axis=0)
        show_lines([*horizontal_lines, *vertical_lines])
        return horizontal_lines, vertical_lines

    def __extract_edges(self, grid: np.ndarray) -> np.ndarray:
        # Smooth the image to remove noise
        grid_blur: np.ndarray = cv.GaussianBlur(grid.copy(), (3, 3), 0)
        if self.show_image:
            cv.imshow('Grid Blur', grid_blur)
            cv.waitKey(0)
            cv.destroyWindow('Grid Blur')

        # Obtain another set of edges using Canny with different params
        edges: np.ndarray = cv.Canny(grid_blur, threshold1=30, threshold2=90, apertureSize=3, L2gradient=True)
        if self.show_image:
            cv.imshow('Canny Edge Detection', edges)
            cv.waitKey(0)
            cv.destroyWindow('Canny Edge Detection')
        return edges


class WindowCellExtractor(CellExtractor):
    def __init__(self,
                 *args,
                 grid_offset: t.Tuple[slice, slice] = (slice(1, None), slice(10, None)),
                 cell_size: t.Tuple[int, int] = (40, 40),
                 stride_offset: t.Tuple[int, int] = (5, 4),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cell_size: np.ndarray = np.array(cell_size)
        self.grid_offset: t.Tuple[slice, slice] = grid_offset
        self.step_size: np.ndarray = np.array([cell_size[0] + stride_offset[0], cell_size[1] + stride_offset[1]])

    def __call__(self, grid: ndarray) -> t.List[t.List[np.ndarray]]:
        cells: np.ndarray = ski.util.shape.view_as_windows(grid[self.grid_offset], self.cell_size.tolist(), self.step_size.tolist())
        cells_as_list: t.List[t.List[np.ndarray]] = []
        for i in range(cells.shape[0]):
            cells_as_list.append([])
            for j in range(cells.shape[1]):
                cells_as_list[-1].append(cells[i][j])
        return cells_as_list


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

