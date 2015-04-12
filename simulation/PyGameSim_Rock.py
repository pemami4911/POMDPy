__author__ = 'patrickemami'

# PyGame
import pygame
from RockModel import RSCellType

# multi-threading
import threading

# BLACK
BACKGROUND = (0, 0, 0)
# WHITE
EMPTY_CELL = (255, 255, 255)
# GREEN
GOAL = (0, 255, 0)
# RED
BAD_ROCKS = (255, 0, 0)
# BLUE
GOOD_ROCKS = (0, 0, 255)

# ORANGE
CURRENT_POS = (255, 128, 0)

# PURPLE
START_POS = (128, 0, 255)

# make grid squares 50 x 50 px
SIZE_X = 50
SIZE_Y = 50

# margin between grid squares
MARGIN = 5

# GLOBAL SHARED VARIABLES
GOOD_ROCKS_LIST = []
GOOD_ROCKS_LOCK = threading.Lock()
# starting agent position
AGENT_POS = None
AGENT_LOCK = threading.Lock()

class Simulator(object):
    """
    This class runs the PyGame simulation of
    """

    def __init__(self, model, solver):

        # initialize pygame
        pygame.init()

        self.model = model
        self.solver = solver
        self.n_rows = self.model.n_rows
        self.n_cols = self.model.n_cols

        self.width = (self.n_cols * SIZE_X) + MARGIN
        self.height = (self.n_rows * SIZE_Y) + MARGIN

        # screen
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("POMDPy Simulator")

        self.exit = False

        self.clock = pygame.time.Clock()

        self.map = self.model.env_map

    # --------- Main Program Loop -------------- #
    def main(self):

        global AGENT_POS
        global GOOD_ROCKS_LIST

        pol = Policy(self.solver)
        pol.start()

        while not self.exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # close
                    self.exit = True

            # draw screen
            self.screen.fill(BACKGROUND)

            # draw rock sample grid
            for i in range(0, self.n_rows):
                for j in range(0, self.n_cols):

                    tile = pygame.Rect(i * SIZE_X, j * SIZE_Y, SIZE_X - MARGIN, SIZE_Y - MARGIN)

                    if (j, i) == AGENT_POS:
                        color = CURRENT_POS

                    elif self.map[j][i] == RSCellType.EMPTY:
                        color = EMPTY_CELL

                    elif self.map[j][i] == RSCellType.GOAL:
                        color = GOAL

                    elif self.map[j][i] >= RSCellType.ROCK:
                        if self.map[j][i] in GOOD_ROCKS_LIST:
                            color = GOOD_ROCKS
                        else:
                            color = BAD_ROCKS

                    pygame.draw.rect(self.screen, color, tile.move(MARGIN, MARGIN))

            pygame.display.flip()
        pygame.quit()


class Policy(threading.Thread):
        """'
        Generates a policy, and then draws it on the grid
        """
        def __init__(self, solver):

            threading.Thread.__init__(self)

            self.solver = solver

        def run(self):

            global AGENT_POS
            global GOOD_ROCKS_LIST

            # TODO
            for policy, total_reward, num_reused_nodes in self.solver.generate_policy():
                print " # -------------- RESET ----------------- # "
                for belief, action, reward in policy:

                    if action.bin_number == 4:
                        print "###############"
                        action.print_action()
                        print "###############"
                    else:
                        action.print_action()

                    good_rocks, bad_rocks = belief.separate_rocks()

                    print "Belief state: Good Rocks: ", good_rocks
                    print "Belief state: Bad Rocks: ", bad_rocks
                    print "Immediate reward: ", reward

                    AGENT_LOCK.acquire()
                    AGENT_POS = (belief.position.i, belief.position.j)
                    AGENT_LOCK.release()

                    GOOD_ROCKS_LOCK.acquire()
                    GOOD_ROCKS_LIST = good_rocks
                    GOOD_ROCKS_LOCK.release()

                    pygame.time.wait(100)

                pygame.time.wait(1000)
if __name__ == '__main__':
    import Solver.Solver
    import RockModel

    model = RockModel.RockModel("RockProblemSim")
    solver = Solver.Solver(model)

    sim = Simulator(model, solver)
    sim.main()

