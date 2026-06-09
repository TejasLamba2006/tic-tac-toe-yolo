import cv2
import numpy as np

class CameraController:
    """Handle camera zoom and pan functionality"""
    def __init__(self, width: int, height: int):
        self.original_width = width
        self.original_height = height
        self.zoom_factor = 1.0  # Default zoom to 1.0x (Normal)
        self.min_zoom = 0.5

        self.max_zoom = 3.0
        self.zoom_step = 0.1
        
        # Pan offsets (default values from requirements)
        self.pan_x = 325  # Default pan x
        self.pan_y = 267   # Default pan y
        self.pan_step = 20
        
        # Flip flag
        self.flip_upside_down = False
        
        # Current viewport
        self.viewport_x = 0
        self.viewport_y = 0
        self.viewport_width = width
        self.viewport_height = height
        
        # Trackbar limits for pan (will be updated based on zoom)
        self.max_pan_x = 0
        self.max_pan_y = 0
        
        self.update_viewport()
    
    def update_viewport(self):
        """Update viewport based on zoom and pan"""
        # Calculate zoomed dimensions
        zoomed_width = int(self.original_width / self.zoom_factor)
        zoomed_height = int(self.original_height / self.zoom_factor)
        
        # Update max pan limits
        self.max_pan_x = max(0, self.original_width - zoomed_width)
        self.max_pan_y = max(0, self.original_height - zoomed_height)
        
        # Apply pan with bounds checking
        self.pan_x = max(0, min(int(self.pan_x), self.max_pan_x))
        self.pan_y = max(0, min(int(self.pan_y), self.max_pan_y))
        
        self.viewport_x = int(self.pan_x)
        self.viewport_y = int(self.pan_y)
        self.viewport_width = min(zoomed_width, self.original_width - self.viewport_x)
        self.viewport_height = min(zoomed_height, self.original_height - self.viewport_y)
    
    def set_pan_x(self, value: int):
        """Set pan X from trackbar"""
        self.pan_x = max(0, min(value, self.max_pan_x))
        self.update_viewport()
    
    def set_pan_y(self, value: int):
        """Set pan Y from trackbar"""
        self.pan_y = max(0, min(value, self.max_pan_y))
        self.update_viewport()
    
    def toggle_flip(self):
        """Toggle upside down flip"""
        self.flip_upside_down = not self.flip_upside_down
        return self.flip_upside_down
    
    def zoom_in(self):
        """Zoom in"""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor = min(self.max_zoom, self.zoom_factor + self.zoom_step)
            self.update_viewport()
            return True
        return False
    
    def zoom_out(self):
        """Zoom out"""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor = max(self.min_zoom, self.zoom_factor - self.zoom_step)
            self.update_viewport()
            return True
        return False
    
    def pan_left(self):
        """Pan left"""
        self.pan_x = max(0, self.pan_x - self.pan_step)
        self.update_viewport()
    
    def pan_right(self):
        """Pan right"""
        self.pan_x = min(self.max_pan_x, self.pan_x + self.pan_step)
        self.update_viewport()
    
    def pan_up(self):
        """Pan up"""
        self.pan_y = max(0, self.pan_y - self.pan_step)
        self.update_viewport()
    
    def pan_down(self):
        """Pan down"""
        self.pan_y = min(self.max_pan_y, self.pan_y + self.pan_step)
        self.update_viewport()
    
    def reset_view(self):
        """Reset zoom and pan to default values"""
        self.zoom_factor = 1.0  # Reset to default zoom
        self.pan_x = 325       # Reset to default pan x

        self.pan_y = 267        # Reset to default pan y
        self.flip_upside_down = False
        self.update_viewport()
    
    def apply_transform(self, frame: np.ndarray) -> np.ndarray:
        """Apply zoom, pan, and flip to frame"""
        if frame is None:
            return None
        
        try:
            # Ensure all coordinates are integers
            start_y = int(self.viewport_y)
            end_y = int(self.viewport_y + self.viewport_height)
            start_x = int(self.viewport_x)
            end_x = int(self.viewport_x + self.viewport_width)
            
            # Bounds checking
            start_y = max(0, min(start_y, frame.shape[0]))
            end_y = max(start_y, min(end_y, frame.shape[0]))
            start_x = max(0, min(start_x, frame.shape[1]))
            end_x = max(start_x, min(end_x, frame.shape[1]))
            
            # Extract viewport region
            cropped = frame[start_y:end_y, start_x:end_x]
            
            # Resize to original dimensions
            if cropped.size > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                result = cv2.resize(cropped, (self.original_width, self.original_height))
                
                # Apply flip if enabled
                if self.flip_upside_down:
                    result = cv2.rotate(result, cv2.ROTATE_180)
                
                return result
            else:
                return frame
                
        except Exception as e:
            print(f"Transform error: {e}")
            return frame
