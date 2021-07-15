import pygame
import random
import neat
import os



pygame.init()
WIDTH = 600
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH,HEIGHT))
WHITE = (255,255,255)
BLACK = (0,0,0)
PIPE_COLOR = (0,255,0)
JUMP_SIZE = 10




def collision_up(bird, pipe):
    dist_x = abs(bird.pos_x - pipe.new_pos_top-pipe.width_top/2);
    dist_y = abs(bird.pos_y - 0-pipe.height_top/2);
    if (dist_x > (pipe.width_top/2 + 10)): return False
    if (dist_y > (pipe.height_top/2 + 10)): return False
    if (dist_x <= (pipe.width_top/2)): return True
    if (dist_y <= (pipe.height_top/2)): return True
    dx=dist_x-pipe.width_top/2;
    dy=dist_y-pipe.width_top/2;
    return (dx*dx+dy*dy<=(100));

def collision_down(bird, pipe):
    dist_x = abs(bird.pos_x - pipe.new_pos_bottom-pipe.width_bottom/2);
    dist_y = abs(bird.pos_y - (HEIGHT-pipe.height_bottom/2));
    if (dist_x > (pipe.width_bottom/2 + 10)): return False
    if (dist_y > (pipe.height_bottom/2 + 10)): return False
    if (dist_x <= (pipe.width_bottom/2)): return True
    if (dist_y <= (pipe.height_bottom/2)): return True
    dx=dist_x-pipe.width_bottom/2;
    dy=dist_y-pipe.width_bottom/2;
    return (dx*dx+dy*dy<=(100));


class Pipe:

    def __init__(self):
        self.width_top = 40
        top_height = random.randint(100,500)
        self.height_top = top_height
        self.width_bottom = 40
        self.height_bottom = HEIGHT - top_height -175
        self.scroller = 0
        self.new_pos_top =  WIDTH - self.width_top
        self.new_pos_bottom = WIDTH - self.width_bottom


    def scroll(self):
        self.scroller += 4
        self.new_pos_top = WIDTH - self.width_top - self.scroller
        self.new_pos_bottom = WIDTH - self.width_bottom - self.scroller
        self.rectangle_top = pygame.draw.rect(screen,PIPE_COLOR,(self.new_pos_top,0,self.width_top,self.height_top))
        self.rectangle_bottom = pygame.draw.rect(screen,BLACK,(self.new_pos_bottom,HEIGHT-self.height_bottom,self.width_bottom,self.height_bottom))


    def update(self,bird):
        if self.new_pos_bottom < bird.pos_x-50:
            bird.fittest = True
            return Pipe(),True

        return self,False



def write_font(bird):
    pygame.font.init()
    myfont = pygame.font.SysFont('Comic Sans MS', 18)
    textsurface = myfont.render(f"Score {bird.score}", False, (0, 0, 0))
    screen.blit(textsurface,(0,0))


class Bird:

    def __init__(self):
        self.pos_x = WIDTH/4
        self.pos_y = HEIGHT/2
        self.is_jump = False
        self.score = 0
        self.fittest = False 
        
    def update(self):
        self.pos_y += 12
        # self.pos_x += 0.1
        if self.fittest:
            self.circle = pygame.draw.circle(screen,BLACK,(self.pos_x,self.pos_y),10)


    def jump(self):
        if self.jump:
            self.pos_y += JUMP_SIZE**2 * -1
            self.is_jump = False

    def at_boundary(self):
        if self.pos_y <= 10 or self.pos_y >= HEIGHT-10 or self.pos_x <= 10 or self.pos_x >= WIDTH-10:
            return True
        return False




# Our fitness function
def main(genomes, config):
    running = True

    screen.fill(WHITE)

    clock = pygame.time.Clock()


    birds = []
    nets = []
    ge = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        g.fitness = 0
        birds.append(Bird())
        ge.append(g)


    pipe = Pipe()

    while running and len(birds) > 0:


        

        jump = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    # jump = True
                    pass
                   


        screen.fill(WHITE)

        for idx , bird in enumerate(birds):
            ge[idx].fitness += 0.1

            output = nets[idx].activate((bird.pos_y/HEIGHT,abs(pipe.height_top/HEIGHT),abs(pipe.height_bottom/HEIGHT),pipe.width_top/WIDTH,pipe.width_bottom/WIDTH))

            if output[0] > 0.5:
                jump = True
            
            if jump:
                bird.is_jump = True
                bird.jump()
                jump = False
            
            bird.update()
        

            pipe, cleared = pipe.update(bird)

            if cleared:
                for g in ge:
                    g.fitness+=10

            if collision_up(bird,pipe) or collision_down(bird,pipe):
                ge[idx].fitness -=50
                birds.pop(idx)
                nets.pop(idx)
                ge.pop(idx)
                bird.score = 0

            if bird.at_boundary():
                ge[idx].fitness -=50
                birds.pop(idx)
                nets.pop(idx)
                ge.pop(idx)
        pipe.scroll()



        # write_font(bird)

        clock.tick(30)


    
        pygame.display.flip()



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main,500)



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-feedforward.txt")
    run(config_path)
    # main()
