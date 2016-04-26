import cv2
import snake
import sys
import math

# Process command line arguments
file_to_load = "example.jpg"
if len(sys.argv) > 1:
    file_to_load = sys.argv[1]

# Loads the desired image
image = cv2.imread( file_to_load, cv2.IMREAD_COLOR )

# Creates the snake
snake = snake.Snake( image, closed = True )

# Window, window name and trackbars
snake_window_name = "Snakes"
controls_window_name = "Controls"
cv2.namedWindow( snake_window_name )
cv2.namedWindow( controls_window_name )
cv2.createTrackbar( "Alpha", controls_window_name, math.floor( snake.alpha * 100 ), 100, snake.set_alpha )
cv2.createTrackbar( "Beta",  controls_window_name, math.floor( snake.beta * 100 ), 100, snake.set_beta )
cv2.createTrackbar( "Delta", controls_window_name, math.floor( snake.delta * 100 ), 100, snake.set_delta )
cv2.createTrackbar( "W Line", controls_window_name, math.floor( snake.w_line * 100 ), 100, snake.set_w_line )
cv2.createTrackbar( "W Edge", controls_window_name, math.floor( snake.w_edge * 100 ), 100, snake.set_w_edge )
cv2.createTrackbar( "W Term", controls_window_name, math.floor( snake.w_term * 100 ), 100, snake.set_w_term )

# Core loop
while( True ):

    # Gets an image of the current state of the snake
    snakeImg = snake.visualize()
    # Shows the image
    cv2.imshow( snake_window_name, snakeImg )
    # Processes a snake step
    snake_changed = snake.step()

    # Stops looping when ESC pressed
    k = cv2.waitKey(33)
    if k == 27:
        break


cv2.destroyAllWindows()
