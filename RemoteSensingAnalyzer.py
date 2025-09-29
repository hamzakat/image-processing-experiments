# Image processing and registeration experiments

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

class RemoteSensingAnalyzer:
    
    def __init__(self, img_path_ref, img_path_sec=None):
        self.ref_bgr = cv2.imread(str(img_path_ref))
        if self.ref_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {img_path_ref}")
        self.ref_gray = cv2.cvtColor(self.ref_bgr, cv2.COLOR_BGR2GRAY)

        if img_path_sec is not None:
            self.sec_bgr = cv2.imread(str(img_path_sec))
            if self.sec_bgr is None:
                raise FileNotFoundError(f"Secondary image not found: {img_path_sec}")
            self.sec_gray = cv2.cvtColor(self.sec_bgr, cv2.COLOR_BGR2GRAY)
        else:
            self.sec_bgr = self.sec_gray = None

        # Store results for analysis
        self.results = {}

    def apply_frequency_domain_filters(self, img, filter_type='gaussian_lowpass', cutoff=50):
        """
        Apply frequency domain filtering techniques
        """
        # Get image dimensions
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2

        # Create frequency domain representation
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)

        # Create filter mask
        if filter_type == 'gaussian_lowpass':
            # Gaussian low-pass filter
            mask = np.zeros((rows, cols), np.uint8)
            x = np.arange(0, cols)
            y = np.arange(0, rows)
            X, Y = np.meshgrid(x, y)
            D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
            mask = np.exp(-(D**2)/(2*(cutoff**2)))

        elif filter_type == 'ideal_lowpass':
            # Ideal low-pass filter
            mask = np.zeros((rows, cols), np.uint8)
            x = np.arange(0, cols)
            y = np.arange(0, rows)
            X, Y = np.meshgrid(x, y)
            D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
            mask[D <= cutoff] = 1

        elif filter_type == 'gaussian_highpass':
            # Gaussian high-pass filter
            mask = np.zeros((rows, cols), np.uint8)
            x = np.arange(0, cols)
            y = np.arange(0, rows)
            X, Y = np.meshgrid(x, y)
            D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
            mask = 1 - np.exp(-(D**2)/(2*(cutoff**2)))

        # Apply filter
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_filtered)

        return img_filtered.astype(np.uint8), mask, f_shift

    def detect_features_comprehensive(self, img, detector_type='sift'):
        """
        Comprehensive feature detection with multiple algorithms
        """
        features = {}
        
        if detector_type == 'sift' or detector_type == 'all':
            try:
                sift = cv2.SIFT_create(nfeatures=2000)
                kp_sift, desc_sift = sift.detectAndCompute(img, None)
                features['SIFT'] = {'keypoints': kp_sift, 'descriptors': desc_sift}
            except:
                print("SIFT not available")

        if detector_type == 'orb' or detector_type == 'all':
            orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=12)
            kp_orb, desc_orb = orb.detectAndCompute(img, None)
            features['ORB'] = {'keypoints': kp_orb, 'descriptors': desc_orb}


        return features

    def robust_feature_matching(self, desc1, desc2, detector_type='sift', ratio_threshold=0.7):
        """
        Robust feature matching with quality metrics
        """
        if desc1 is None or desc2 is None:
            return [], {}

        if detector_type == 'SIFT':
            norm = cv2.NORM_L2
        else:
            norm = cv2.NORM_HAMMING

        bf = cv2.BFMatcher(norm)
        raw_matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test and collect statistics
        good_matches = []
        distances = []
        ratios = []
        
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                ratio = m.distance / n.distance
                ratios.append(ratio)
                if ratio < ratio_threshold:
                    good_matches.append(m)
                    distances.append(m.distance)
        
        match_stats = {
            'total_raw_matches': len(raw_matches),
            'good_matches': len(good_matches),
            'mean_distance': np.mean(distances) if distances else 0,
            'std_distance': np.std(distances) if distances else 0,
            'mean_ratio': np.mean(ratios) if ratios else 0
        }
        
        return good_matches, match_stats

    def estimate_homography_with_analysis(self, img_ref, img_sec, detector='sift'):
        """
        Homography estimation with comprehensive analysis
        """
        # Feature detection
        features_ref = self.detect_features_comprehensive(img_ref, detector)
        features_sec = self.detect_features_comprehensive(img_sec, detector)
        
        detector_key = detector.upper() if detector != 'all' else 'SIFT'
        
        if detector_key not in features_ref or detector_key not in features_sec:
            return None, None, None, None, {}
        
        kp1 = features_ref[detector_key]['keypoints']
        desc1 = features_ref[detector_key]['descriptors']
        kp2 = features_sec[detector_key]['keypoints']
        desc2 = features_sec[detector_key]['descriptors']
        
        # Feature matching
        matches, match_stats = self.robust_feature_matching(desc1, desc2, detector_key)
        
        if len(matches) < 10:
            return None, kp1, kp2, matches, match_stats
        
        # Extract points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # RANSAC homography estimation
        H, mask = cv2.findHomography(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.995
        )
        
        inliers = [matches[i] for i, m in enumerate(mask.ravel()) if m] if mask is not None else []
        
        # Add homography analysis
        match_stats.update({
            'inliers': len(inliers),
            'inlier_ratio': len(inliers) / len(matches) if matches else 0,
            'homography_found': H is not None
        })
        
        return H, kp1, kp2, inliers, match_stats

    def visualize_feature_detection(self, img, features, title="Feature Detection", save_path=None):
        """
        Create publication-ready feature detection visualization with adaptive layout
        """
        detector_names = list(features.keys())
        n_detectors = len(detector_names)
        
        # Adaptive layout based on number of detectors
        if n_detectors == 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes = axes.flatten()
        elif n_detectors == 2:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Original image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Feature detection results
        for i, detector_name in enumerate(detector_names):
            if i + 1 < len(axes):
                ax = axes[i + 1]
                kps = features[detector_name]['keypoints']
                
                # Draw image with keypoints
                img_with_kp = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)
                ax.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
                ax.set_title(f'{detector_name} Features ({len(kps)} points)')
                ax.axis('off')
        
        # Hide unused axes
        for i in range(n_detectors + 1, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_frequency_analysis(self, original, filtered_results, save_path=None):
        """
        Visualize frequency domain analysis
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        filters = ['gaussian_lowpass', 'ideal_lowpass', 'gaussian_highpass']
        cutoffs = [30, 50, 30]
        
        for i, (filter_type, cutoff) in enumerate(zip(filters, cutoffs)):
            filtered_img, mask, f_shift = self.apply_frequency_domain_filters(
                original, filter_type, cutoff
            )
            
            # Filtered image
            axes[i,0].imshow(filtered_img, cmap='gray')
            axes[i,0].set_title(f'{filter_type.replace("_", " ").title()}')
            axes[i,0].axis('off')
            
            # Filter mask
            axes[i,1].imshow(mask, cmap='gray')
            axes[i,1].set_title(f'Filter Mask (cutoff={cutoff})')
            axes[i,1].axis('off')
            
            # Frequency spectrum
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            axes[i,2].imshow(magnitude_spectrum, cmap='hot')
            axes[i,2].set_title('Frequency Spectrum')
            axes[i,2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_matches_annotated(self, img1, img2, kp1, kp2, matches, title="Feature Matches", save_path=None):
        """
        Create annotated match visualization with close-up regions
        """
        # Main match visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Full image matches
        ax_main = plt.subplot(2, 3, (1, 3))
        
        # Convert images to RGB for matplotlib
        if len(img1.shape) == 3:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        else:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        # Create side-by-side image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        combined_img[:h1, :w1] = img1_rgb
        combined_img[:h2, w1:w1+w2] = img2_rgb
        
        ax_main.imshow(combined_img)
        
        # Draw matches with different colors based on quality
        match_distances = [m.distance for m in matches]
        if match_distances:
            dist_threshold = np.percentile(match_distances, 75)
            
            for i, match in enumerate(matches[:50]):  # Limit for clarity
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                pt2_shifted = (pt2[0] + w1, pt2[1])
                
                # Color based on match quality
                color = 'green' if match.distance < dist_threshold else 'orange'
                linewidth = 2 if match.distance < dist_threshold else 1
                
                ax_main.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], 
                           color=color, linewidth=linewidth, alpha=0.7)
                
                # Mark keypoints
                ax_main.scatter(pt1[0], pt1[1], c='red', s=30, marker='o')
                ax_main.scatter(pt2_shifted[0], pt2_shifted[1], c='blue', s=30, marker='o')
        
        ax_main.set_title(f'{title} (Showing best matches)', fontsize=14)
        ax_main.axis('off')
        
        # Close-up regions showing match quality
        if len(matches) > 0:
            # Select best matches for close-ups
            sorted_matches = sorted(matches, key=lambda x: x.distance)
            
            for i, match_idx in enumerate([0, len(sorted_matches)//4, len(sorted_matches)//2]):
                if match_idx < len(sorted_matches):
                    match = sorted_matches[match_idx]
                    pt1 = kp1[match.queryIdx].pt
                    pt2 = kp2[match.trainIdx].pt
                    
                    # Close-up from reference image
                    ax_close = plt.subplot(2, 3, 4 + i)
                    
                    # Extract region around keypoint
                    x, y = int(pt1[0]), int(pt1[1])
                    size = 60
                    x1, y1 = max(0, x-size), max(0, y-size)
                    x2, y2 = min(img1.shape[1], x+size), min(img1.shape[0], y+size)
                    
                    region = img1_rgb[y1:y2, x1:x2]
                    ax_close.imshow(region)
                    
                    # Mark the keypoint
                    center_x, center_y = x - x1, y - y1
                    circle = Circle((center_x, center_y), 5, color='red', fill=False, linewidth=2)
                    ax_close.add_patch(circle)
                    
                    # Add orientation line if available
                    if hasattr(kp1[match.queryIdx], 'angle') and kp1[match.queryIdx].angle != -1:
                        angle = np.radians(kp1[match.queryIdx].angle)
                        dx = 15 * np.cos(angle)
                        dy = 15 * np.sin(angle)
                        ax_close.arrow(center_x, center_y, dx, dy, 
                                     head_width=3, head_length=3, fc='yellow', ec='yellow')
                    
                    ax_close.set_title(f'Match Quality: {match.distance:.2f}')
                    ax_close.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_analysis_table(self, results_dict, save_path=None):
        """
        Create comprehensive analysis table
        """
        # Convert results to DataFrame
        df_data = []
        for detector, stats in results_dict.items():
            df_data.append({
                'Detector': detector,
                'Raw Matches': stats.get('total_raw_matches', 0),
                'Good Matches': stats.get('good_matches', 0),
                'Inliers': stats.get('inliers', 0),
                'Inlier Ratio (%)': f"{stats.get('inlier_ratio', 0)*100:.1f}",
                'Mean Distance': f"{stats.get('mean_distance', 0):.2f}",
                'Mean Ratio': f"{stats.get('mean_ratio', 0):.3f}",
                'Homography': 'Yes' if stats.get('homography_found', False) else 'No'
            })
        
        df = pd.DataFrame(df_data)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Feature Detection and Matching Analysis', fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df

    def comprehensive_analysis(self, detector_types=['sift', 'orb']):
        """
        Perform comprehensive analysis suitable for essay inclusion
        """
        if self.sec_gray is None:
            print("Second image required for comprehensive analysis")
            return
        
        print("=== Comprehensive Remote Sensing Analysis ===\n")
        
        # 1. Frequency Domain Analysis
        print("1. Analyzing frequency domain filtering effects...")
        self.visualize_frequency_analysis(
            self.ref_gray, 
            None,  # Will be computed inside the function
            save_path="frequency_analysis.png"
        )
        
        # 2. Feature Detection Analysis
        print("2. Analyzing feature detection algorithms...")
        features_ref = self.detect_features_comprehensive(self.ref_gray, 'all')
        self.visualize_feature_detection(
            self.ref_gray, 
            features_ref, 
            title="Feature Detection Comparison - Reference Image",
            save_path="feature_detection_comparison.png"
        )
        
        # 3. Feature Matching Analysis
        print("3. Analyzing feature matching performance...")
        results = {}
        
        for detector in detector_types:
            print(f"\nAnalyzing {detector.upper()} detector...")
            H, kp1, kp2, matches, stats = self.estimate_homography_with_analysis(
                self.ref_gray, self.sec_gray, detector
            )
            
            results[detector.upper()] = stats
            
            if H is not None and len(matches) > 0:
                # Create detailed match visualization
                self.visualize_matches_annotated(
                    self.ref_gray, self.sec_gray, kp1, kp2, matches,
                    title=f"{detector.upper()} Feature Matches",
                    save_path=f"{detector}_matches_annotated.png"
                )
                
                print(f"  - Found {len(matches)} inliers")
                print(f"  - Inlier ratio: {stats['inlier_ratio']*100:.1f}%")
                print(f"  - Mean match distance: {stats['mean_distance']:.2f}")
        
        # 4. Create comprehensive analysis table
        print("4. Creating analysis summary...")
        df_results = self.create_analysis_table(
            results, 
            save_path="analysis_summary_table.png"
        )
        
        # 5. Save results for essay
        df_results.to_csv("feature_analysis_results.csv", index=False)
        print("\nAnalysis complete! Generated files:")
        print("  - frequency_analysis.png")
        print("  - feature_detection_comparison.png")
        print("  - *_matches_annotated.png")
        print("  - analysis_summary_table.png")
        print("  - feature_analysis_results.csv")
        
        return results, df_results

# Example usage
if __name__ == "__main__":
    ref_img = Path("city_aerial_image.jpg")
    zoom_img = Path("city_aerial_image_zoomed.jpg")
    
    if ref_img.exists() and zoom_img.exists():
        analyzer = RemoteSensingAnalyzer(ref_img, zoom_img)
        results, summary_df = analyzer.comprehensive_analysis(['sift', 'orb'])
        
        print("\n=== Summary Statistics ===")
        print(summary_df.to_string(index=False))
        
    else:
        print("Image files not found. Please ensure you have:")
        print("  - city_aerial_image.jpg (reference)")
        print("  - city_aerial_image_zoomed.jpg (test)")