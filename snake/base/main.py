import pygame as pg
from game_objects import Snake, Food
import sys


class Game:
    @property
    def WINDOW_SIZE(self):
        return 1000

    @property
    def TILE_SIZE(self):
        return 50

    def __init__(self) -> None:
        pg.init()

        # Inicializar a tela
        self.screen = pg.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        self.clock = pg.time.Clock()
        self.new_game()

    def draw_grid(self):
        # Desenhar a grade
        # vertical lines
        [
            pg.draw.line(
                surface=self.screen,
                color=(50, 50, 50),
                start_pos=(x, 0),
                end_pos=(x, self.WINDOW_SIZE),
            )
            for x in range(0, self.WINDOW_SIZE, self.TILE_SIZE)
        ]

        # horizontal lines
        [
            pg.draw.line(
                surface=self.screen,
                color=(50, 50, 50),
                start_pos=(0, y),
                end_pos=(self.WINDOW_SIZE, y),
            )
            for y in range(0, self.WINDOW_SIZE, self.TILE_SIZE)
        ]

    def new_game(self):
        self.snake = Snake(self)
        self.food = Food(self)

    def update(self):
        self.snake.update()  # update the snake
        pg.display.flip()  # update the screen
        self.clock.tick(60)  # 60 FPS

    def draw(self):
        # Desenhar a tela
        self.screen.fill((0, 0, 0))
        self.draw_grid()
        self.food.draw()
        self.snake.draw()

    def check_event(self):
        # Checar eventos
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            # snake control
            self.snake.handle_events(event)
            

    def run(self):
        # game loop
        while True:
            self.check_event()
            self.update()
            self.draw()


if __name__ == "__main__":
    game = Game()
    game.run()
