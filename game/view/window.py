import pygame


class Window:
    def __init__(self):
        self.width = 800
        self.height = 600

        self.surface = pygame.display.set_mode((self.width, self.height))

    def draw(self, character, ball):
        self.surface.fill((0, 0, 0))

        character.draw(self.surface)
        ball.draw(self.surface)

        pygame.display.update()