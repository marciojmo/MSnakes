import numpy as np
import cv2
import math


class Snake:
    """ A Snake class for active contour segmentation """

    # Constants
    MIN_DISTANCE_BETWEEN_POINTS = 5    # The minimum distance between two points to consider them overlaped
    MAX_DISTANCE_BETWEEN_POINTS = 50    # The maximum distance to insert another point into the spline
    SEARCH_KERNEL_SIZE = 7              # The size of the search kernel.

    # Members
    image = None        # The source image.
    gray = None         # The image in grayscale.
    binary = None       # The image in binary (threshold method).
    gradientX = None    # The gradient (sobel) of the image relative to x.
    gradientY = None    # The gradient (sobel) of the image relative to y.
    blur = None
    width = -1          # The image width.
    height = -1         # The image height.
    points = None       # The list of points of the snake.
    n_starting_points = 50       # The number of starting points of the snake.
    snake_length = 0    # The length of the snake (euclidean distances).
    closed = True       # Indicates if the snake is closed or open.
    alpha = 0.5         # The weight of the uniformity energy.
    beta = 0.5          # The weight of the curvature energy.
    delta = 0.1         # The weight of the user configured energy.
    w_line = 0.5        # The weight to the line energy.
    w_edge = 0.5        # The weight to the edge energy.
    w_term = 0.5        # The weight to the term energy.




    #################################
    # Constructor
    #################################
    def __init__( self, image = None, closed = True ):
        """
        Object constructor
        :param image: The image to run snake on
        :return:
        """

        # Sets the image and it's properties
        self.image = image

        # Image properties
        self.width = image.shape[1]
        self.height = image.shape[0]

        # Image variations used by the snake
        self.gray = cv2.cvtColor( self.image, cv2.COLOR_RGB2GRAY )
        self.binary = cv2.adaptiveThreshold( self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2 )
        self.gradientX = cv2.Sobel( self.gray, cv2.CV_64F, 1, 0, ksize=5 )
        self.gradientY = cv2.Sobel( self.gray, cv2.CV_64F, 0, 1, ksize=5 )

        # Set the snake behaviour (closed or open)
        self.closed = closed

        # Gets the half width and height of the image
        # to use as center of the generated snake circle
        half_width = math.floor( self.width / 2 )
        half_height = math.floor( self.height / 2 )


        ####################################
        # Generating the starting points   #
        ####################################
        # If it is a closed snake, my initial guess will be a circle
        # that covers the maximum of the image
        if self.closed:
            n = self.n_starting_points
            radius = half_width if half_width < half_height else half_height
            self.points = [ np.array([
                half_width + math.floor( math.cos( 2 * math.pi / n * x ) * radius ),
                half_height + math.floor( math.sin( 2 * math.pi / n * x ) * radius ) ])
                for x in range( 0, n )
            ]
        else:   # If it is an open snake, the initial guess will be an horizontal line
            n = self.n_starting_points
            factor = math.floor( half_width / (self.n_starting_points-1) )
            self.points = [ np.array([ math.floor( half_width / 2 ) + x * factor, half_height ])
                for x in range( 0, n )
            ]



    def visualize( self ):
        """
        Draws the current state of the snake over the image.
        :return: An image with the snake drawed over it
        """
        img = self.image.copy()

        # Drawing lines between points
        point_color = ( 0, 255, 255 )     # BGR RED
        line_color = ( 128, 0, 0 )      # BGR half blue
        thickness = 2                   # Thickness of the lines and circles

        # Draw a line between the current and the next point
        n_points = len( self.points )
        for i in range( 0, n_points - 1 ):
            cv2.line( img, tuple( self.points[ i ] ), tuple( self.points[ i + 1 ] ), line_color, thickness )

        # 0 -> N (Closes the snake)
        if self.closed:
            cv2.line(img, tuple( self.points[ 0 ] ), tuple( self.points[ n_points-1 ] ), line_color, thickness )

        # Drawing circles over points
        [ cv2.circle( img, tuple( x ), thickness, point_color, -1) for x in self.points ]

        return img



    def dist( a, b ):
        """
        Calculates the euclidean distance between two points
        :param a: The first point
        :param b: The second point
        :return: The euclidian distance between the two points
        """

        return np.sqrt( np.sum( ( a - b ) ** 2 ) )



    def normalize( kernel ):
        """
        Normalizes a kernel
        :param kernel: The kernel full of numbers to normalize
        :return: A copy of the kernel normalized
        """

        abs_sum = np.sum( [ abs( x ) for x in kernel ] )
        return kernel / abs_sum if abs_sum != 0 else kernel



    def get_length(self):
        """
        The length of the snake (euclidean distances)
        :return: The lenght of the snake (euclidean distances)
        """

        n_points = len(self.points)
        if not self.closed:
            n_points -= 1

        return np.sum( [ Snake.dist( self.points[i], self.points[ (i+1)%n_points  ] ) for i in range( 0, n_points ) ] )



    def f_uniformity( self, p, prev ):
        """
        The uniformity energy. (The tendency to move the curve towards a straight line)
        :param p: The point being analysed
        :param prev: The previous point in the curve
        :return: The uniformity energy for the given points
        """
        # The average distance between points in the snake
        avg_dist = self.snake_length / len( self.points )
        # The distance between the previous and the point being analysed
        un = Snake.dist( prev, p )

        dun = abs( un - avg_dist )

        return dun**2



    def f_curvature( self, p, prev, next ):
        """
        The Curvature energy
        :param p: The point being analysed
        :param prev: The previous point in the curve
        :param next: The next point in the curve
        :return: The curvature energy for the given points
        """
        ux = p[0] - prev[0]
        uy = p[1] - prev[1]
        un = math.sqrt( ux**2 + uy**2 )

        vx = p[0] - next[0]
        vy = p[1] - next[1]
        vn = math.sqrt( vx**2 + vy**2 )

        if un == 0 or vn == 0:
            return 0

        cx = float( vx + ux )  / ( un * vn )
        cy = float( vy + uy ) / ( un * vn )

        cn = cx**2 + cy**2

        return cn



    def f_line( self, p ):
        """
        The line energy (The tendency to move the curve towards dark / lighter areas)
        :param p: The point being analysed
        :return: The line energy for the given point
        """
        # If the point is out of the bounds of the image, return a high value
        # (since it is a minimization problem)
        if p[0] < 0 or p[0] >= self.width or p[1] < 0 or p[1] >= self.height:
            return np.finfo(np.float64).max

        return self.binary[ p[1] ][ p[0] ]



    def f_edge( self, p ):
        """
        The edge energy (The tendency to move the curve towards edges). Using the sobel gradient.
        :param p: The point being analysed
        :return: The edge energy for the given point
        """
        # If the point is out of the bounds of the image, return a high value
        # (since it is a minimization problem)
        if p[0] < 0 or p[0] >= self.width or p[1] < 0 or p[1] >= self.height:
            return np.finfo(np.float64).max

        return -( self.gradientX[ p[1] ][ p[0] ]**2 + self.gradientY[ p[1] ][ p[0] ]**2  )



    def f_term( self, p, prev, next ):
        """
        Not implemented. The tendency to move the snake towards corners and terminations.
        :param p: The point being analysed
        :return: The term energy.
        """
        return 0



    def f_conf( self, p , prev, next ):
        """
        User configurable energy. The tendency to do whatever you want.
        In this case, I'm adding a perturbation tendency on each point.
        :param p: The point being analysed
        :param prev: The previous point on the snake
        :param next: The next point on the snake
        :return: The user configurable energy
        """
        import random
        return random.random()



    def remove_overlaping_points( self ):
        """
        Remove overlaping points from the curve based on
        the minimum distance between points (MIN_DIST_BETWEEN_POINTS)
        """

        snake_size = len( self.points )

        for i in range( 0, snake_size ):
            for j in range( snake_size-1, i+1, -1 ):
                if i == j:
                    continue

                curr = self.points[ i ]
                end = self.points[ j ]

                dist = Snake.dist( curr, end )

                if dist < self.MIN_DISTANCE_BETWEEN_POINTS:
                    remove_indexes = range( i+1, j ) if (i!=0 and j!=snake_size-1) else [j]
                    remove_size = len( remove_indexes )
                    non_remove_size = snake_size - remove_size
                    if non_remove_size > remove_size:
                        self.points = [ p for k,p in enumerate( self.points ) if k not in remove_indexes ]
                    else:
                        self.points = [ p for k,p in enumerate( self.points ) if k in remove_indexes ]
                    snake_size = len( self.points )
                    break





    def add_missing_points( self ):
        """
        Add points to the spline if the distance between two points is bigger than
        the maximum distance (MAX_DISTANCE_BETWEEN_POINTS)
        """
        snake_size = len( self.points )
        for i in range( 0, snake_size ):
            prev = self.points[ ( i + snake_size-1 ) % snake_size ]
            curr = self.points[ i ]
            next = self.points[ (i+1) % snake_size ]
            next2 = self.points[ (i+2) % snake_size ]

            if Snake.dist( curr, next ) > self.MAX_DISTANCE_BETWEEN_POINTS:
                # Pre-computed uniform cubig b-spline for t = 0.5
                c0 = 0.125 / 6.0
                c1 = 2.875 / 6.0
                c2 = 2.875 / 6.0
                c3 = 0.125 / 6.0
                x = prev[0] * c3 + curr[0] * c2 + next[0] * c1 + next2[0] * c0
                y = prev[1] * c3 + curr[1] * c2 + next[1] * c1 + next2[1] * c0

                new_point = np.array( [ math.floor( 0.5 + x ), math.floor( 0.5 + y ) ] )

                self.points.insert( i+1, new_point )
                snake_size += 1




    def step( self ):
        """
        Perform a step in the active contour algorithm
        """
        changed = False

        # Computes the length of the snake (used by uniformity function)
        self.snake_length = self.get_length()
        new_snake = self.points.copy()


        # Kernels (They store the energy for each point being search along the search kernel)
        search_kernel_size = ( self.SEARCH_KERNEL_SIZE, self.SEARCH_KERNEL_SIZE )
        hks = math.floor( self.SEARCH_KERNEL_SIZE / 2 ) # half-kernel size
        e_uniformity = np.zeros( search_kernel_size )
        e_curvature = np.zeros( search_kernel_size )
        e_line = np.zeros( search_kernel_size )
        e_edge = np.zeros( search_kernel_size )
        e_term = np.zeros( search_kernel_size )
        e_conf = np.zeros( search_kernel_size )

        for i in range( 0, len( self.points ) ):
            curr = self.points[ i ]
            prev = self.points[ ( i + len( self.points )-1 ) % len( self.points ) ]
            next = self.points[ ( i + 1 ) % len( self.points ) ]


            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    p = np.array( [curr[0] + dx, curr[1] + dy] )

                    # Calculates the energy functions on p
                    e_uniformity[ dx + hks ][ dy + hks ] = self.f_uniformity( p, prev )
                    e_curvature[ dx + hks ][ dy + hks ] = self.f_curvature( p, prev, next )
                    e_line[ dx + hks ][ dy + hks ] = self.f_line( p )
                    e_edge[ dx + hks ][ dy + hks ] = self.f_edge( p )
                    e_term[ dx + hks ][ dy + hks ] = self.f_term( p, prev, next )
                    e_conf[ dx + hks ][ dy + hks ] = self.f_conf( p, prev, next )


            # Normalizes energies
            e_uniformity = Snake.normalize( e_uniformity )
            e_curvature = Snake.normalize( e_curvature )
            e_line = Snake.normalize( e_line )
            e_edge = Snake.normalize( e_edge )
            e_term = Snake.normalize( e_term )
            e_conf = Snake.normalize( e_conf )



            # The sum of all energies for each point

            e_sum = self.alpha * e_uniformity \
                    + self.beta * e_curvature \
                    + self.w_line * e_line \
                    + self.w_edge * e_edge \
                    + self.w_term * e_term \
                    + self.delta * e_conf

            # Searches for the point that minimizes the sum of energies e_sum
            emin = np.finfo(np.float64).max
            x,y = 0,0
            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    if e_sum[ dx + hks ][ dy + hks ] < emin:
                        emin = e_sum[ dx + hks ][ dy + hks ]
                        x = curr[0] + dx
                        y = curr[1] + dy

            # Boundary check
            x = 1 if x < 1 else x
            x = self.width-2 if x >= self.width-1 else x
            y = 1 if y < 1 else y
            y = self.height-2 if y >= self.height-1 else y

            # Check for changes
            if curr[0] != x or curr[1] != y:
                changed = True

            new_snake[i] = np.array( [ x, y ] )

        self.points = new_snake

        # Post threatment to the snake, remove overlaping points and
        # add missing points
        self.remove_overlaping_points()
        self.add_missing_points()

        return changed



    def set_alpha( self, x ):
        """
        Utility function (used by cvCreateTrackbar) to set the value of alpha.
        :param x: The new value of alpha (scaled by 100)
        """
        self.alpha = x / 100



    def set_beta( self, x ):
        """
        Utility function (used by cvCreateTrackbar) to set the value of beta.
        :param x: The new value of beta (scaled by 100)
        """
        self.beta = x / 100



    def set_delta( self, x ):
        """
        Utility function (used by cvCreateTrackbar) to set the value of delta.
        :param x: The new value of delta (scaled by 100)
        """
        self.delta = x / 100



    def set_w_line( self, x ):
        """
        Utility function (used by cvCreateTrackbar) to set the value of w_line.
        :param x: The new value of w_line (scaled by 100)
        """
        self.w_line = x / 100



    def set_w_edge( self, x ):
        """
        Utility function (used by cvCreateTrackbar) to set the value of w_edge.
        :param x: The new value of w_edge (scaled by 100)
        """
        self.w_edge = x / 100



    def set_w_term( self, x ):
        """
        Utility function (used by cvCreateTrackbar) to set the value of w_term.
        :param x: The new value of w_term (scaled by 100)
        """
        self.w_term = x / 100