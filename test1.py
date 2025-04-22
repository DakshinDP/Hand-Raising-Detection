import pygame
import pygame.camera

# Initialize Pygame camera
pygame.camera.init()

# List available cameras
cameras = pygame.camera.get_camera()

if cameras:
    print(f"Available camera: {cameras[0]}")
    cam = pygame.camera.Camera(cameras[0], (640, 480))
    cam.start()
else:
    print("No camera detected.")

# Capture an image to see if it works
image = cam.get_image()
pygame.image.save(image, "test_image.jpg")

# Stop camera
cam.stop()
