import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import time

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Webcam Fruit Ninja')
clock = pygame.time.Clock()

# Initialize OpenCV webcam
cap = cv2.VideoCapture(0)

# Mediapipe hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Fruit class and helper
class Fruit:
    def __init__(self):
        self.radius = random.randint(25, 40)
        self.x = random.randint(self.radius, WIDTH - self.radius)
        self.y = HEIGHT + self.radius
        self.color = random.choice([(255,0,0), (0,255,0), (255,255,0), (255,140,0), (128,0,128)])
        self.vel_y = -random.uniform(20, 28)
        self.gravity = 0.5
        self.active = True
    def update(self):
        self.y += self.vel_y
        self.vel_y += self.gravity
        if self.y - self.radius > HEIGHT:
            self.active = False
    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

# Game variables
running = True
fruits = []
fruit_spawn_timer = 0
FRUIT_SPAWN_INTERVAL = 40  # frames
score = 0
finger_trail = []
MAX_TRAIL_LENGTH = 15
prev_tip = None
prev_time = None
SWIPE_SPEED_THRESHOLD = 30  # pixels per frame
ROUND_TIME = 40  # seconds
start_time = time.time()
game_over = False

def reset_game():
    global fruits, fruit_spawn_timer, score, finger_trail, prev_tip, start_time, game_over, running
    fruits = []
    fruit_spawn_timer = 0
    score = 0
    finger_trail = []
    prev_tip = None
    start_time = time.time()
    game_over = False
    running = True

# Main loop
while True:
    reset_game()
    while running:
        # --- Timer logic ---
        elapsed = time.time() - start_time
        time_left = max(0, int(ROUND_TIME - elapsed))
        if time_left == 0 and not game_over:
            game_over = True
            running = False

        # --- Handle Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                exit()
            if game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                running = False  # Will trigger reset in outer loop

        # --- Webcam frame capture ---
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Horizontal flip for mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Hand tracking ---
        results = hands.process(rgb_frame)
        tip_pos = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Get index finger tip (landmark 8)
                h, w, _ = frame.shape
                tip = hand_landmarks.landmark[8]
                # Mirror the x coordinate for slicing logic
                mirrored_x = w - int(tip.x * w)
                tip_pos = (mirrored_x, int(tip.y * h))
                finger_trail.append(tip_pos)
                if len(finger_trail) > MAX_TRAIL_LENGTH:
                    finger_trail.pop(0)
        else:
            finger_trail.clear()

        # --- Slicing detection ---
        if len(finger_trail) > 1:
            for i in range(len(finger_trail) - 1):
                x1, y1 = finger_trail[i]
                x2, y2 = finger_trail[i+1]
                for fruit in fruits:
                    if fruit.active:
                        # Distance from fruit center to line segment
                        px, py = fruit.x, fruit.y
                        dx, dy = x2 - x1, y2 - y1
                        if dx == dy == 0:
                            dist = ((px - x1)**2 + (py - y1)**2) ** 0.5
                        else:
                            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
                            proj_x = x1 + t * dx
                            proj_y = y1 + t * dy
                            dist = ((px - proj_x)**2 + (py - proj_y)**2) ** 0.5
                        if dist < fruit.radius:
                            fruit.active = False
                            score += 1

        # --- Game logic ---
        fruit_spawn_timer += 1
        if fruit_spawn_timer >= FRUIT_SPAWN_INTERVAL:
            fruits.append(Fruit())
            fruit_spawn_timer = 0
        for fruit in fruits:
            fruit.update()
        fruits = [f for f in fruits if f.active]

        # --- Drawing ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        frame_surface = pygame.transform.scale(frame_surface, (WIDTH, HEIGHT))
        screen.blit(frame_surface, (0, 0))
        for fruit in fruits:
            fruit.draw(screen)
        # Draw slicing trail
        if len(finger_trail) > 1:
            pygame.draw.lines(screen, (0,255,255), False, [(int(x), int(y)) for (x,y) in finger_trail], 5)
        # Draw score
        font = pygame.font.SysFont(None, 48)
        score_surf = font.render(f'Score: {score}', True, (255,255,255))
        screen.blit(score_surf, (10, 10))
        # Draw timer
        timer_surf = font.render(f'Time: {time_left}', True, (255,255,0))
        screen.blit(timer_surf, (WIDTH - 180, 10))
        # Game over message
        if game_over:
            over_font = pygame.font.SysFont(None, 72)
            over_surf = over_font.render('Game Over!', True, (255,0,0))
            screen.blit(over_surf, (WIDTH//2 - over_surf.get_width()//2, HEIGHT//2 - 60))
            final_score = font.render(f'Final Score: {score}', True, (255,255,255))
            screen.blit(final_score, (WIDTH//2 - final_score.get_width()//2, HEIGHT//2 + 10))
            restart_surf = font.render('Press R to Restart', True, (0,255,0))
            screen.blit(restart_surf, (WIDTH//2 - restart_surf.get_width()//2, HEIGHT//2 + 60))
        pygame.display.flip()
        clock.tick(60)

    # Game over screen loop
    while game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game_over = False
        # Draw final frame with game over and restart message
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        frame_surface = pygame.transform.scale(frame_surface, (WIDTH, HEIGHT))
        screen.blit(frame_surface, (0, 0))
        for fruit in fruits:
            fruit.draw(screen)
        if len(finger_trail) > 1:
            pygame.draw.lines(screen, (0,255,255), False, [(int(x), int(y)) for (x,y) in finger_trail], 5)
        font = pygame.font.SysFont(None, 48)
        score_surf = font.render(f'Score: {score}', True, (255,255,255))
        screen.blit(score_surf, (10, 10))
        timer_surf = font.render(f'Time: 0', True, (255,255,0))
        screen.blit(timer_surf, (WIDTH - 180, 10))
        over_font = pygame.font.SysFont(None, 72)
        over_surf = over_font.render('Game Over!', True, (255,0,0))
        screen.blit(over_surf, (WIDTH//2 - over_surf.get_width()//2, HEIGHT//2 - 60))
        final_score = font.render(f'Final Score: {score}', True, (255,255,255))
        screen.blit(final_score, (WIDTH//2 - final_score.get_width()//2, HEIGHT//2 + 10))
        restart_surf = font.render('Press R to Restart', True, (0,255,0))
        screen.blit(restart_surf, (WIDTH//2 - restart_surf.get_width()//2, HEIGHT//2 + 60))
        pygame.display.flip()
        clock.tick(60)

# Cleanup
cap.release()
pygame.quit() 