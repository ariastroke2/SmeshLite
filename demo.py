"""
SmeshLite — player-controlled sandbox.

Controls (4-button scheme, matching original game):
  P1: A/D = left/right   W = jump   S = attack / charge
  P2: ←/→ = left/right   ↑ = jump   ↓ = attack / charge

  ESC   = pause / resume

Pause menu:
  ESC / Space / Enter  → Resume
  R                    → Restart
  Q                    → Quit
"""
import pygame
from smeshlite.core.match import Match, MatchConfig
from smeshlite.core.brain import PlayerInput
from smeshlite.render.renderer import Renderer, SCREEN_W, SCREEN_H

# Importa tus agentes aquí
from agents  import chaser_bot, random_bot, smesh_bot, qtable_bot, deepq_bot_v2


# Build match and wire PlayerInput brains
config = MatchConfig(stocks=3, time_limit=7200)
match = Match(config)
match.reset(n_players=2)

# Definir como usarás los cerebros para los jugadores aquí!
def demo_character_brains_config(match):
    # Fijar cerebro del jugador 1 al input WASD
    # match.characters[0].set_brain(PlayerInput(
    #     key_left=pygame.K_a,
    #     key_right=pygame.K_d,
    #     key_up=pygame.K_w,
    #     key_attack=pygame.K_s,
    # ))

    match.characters[0].set_brain(
        qtable_bot.QTableBot(q_table_path="agents/checkpoints/qtable_bot_roundrobin.json", epsilon=0.0)
    )

    match.characters[1].set_brain(
        deepq_bot_v2.DeepQBot(checkpoint_path="agents/checkpoints/deepq_bot.pt", epsilon=0.0)
    )

    # match.characters[0].set_brain(smesh_bot.SmeshBot())

    # Fijar cerebro del jugador 2 al input ARROW KEYS
    # match.characters[1].set_brain(PlayerInput(
    #     key_left=pygame.K_LEFT,
    #     key_right=pygame.K_RIGHT,
    #     key_up=pygame.K_UP,
    #     key_attack=pygame.K_DOWN,
    # ))

    # Fijar cerebro del jugador a política RANDOM
    # match.characters[1].set_brain(random_bot.RandomBot())

    # Fijar cerebro del jugador a política de persecución simple
    # match.characters[1].set_brain(chaser_bot.ChaserBot())

demo_character_brains_config(match)

renderer = Renderer(render_mode="human")
renderer.render(match)   # force pygame init before event loop

font_title = pygame.font.SysFont("monospace", 38, bold=True)
font_item  = pygame.font.SysFont("monospace", 22)

print("P1: A/D=move  W=jump  S=attack/charge")
print("P2: ←/→=move  ↑=jump  ↓=attack/charge")


def draw_pause_overlay() -> None:
    screen  = pygame.display.get_surface()
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    cx = SCREEN_W // 2
    cy = SCREEN_H // 2

    title = font_title.render("PAUSED", True, (255, 255, 255))
    screen.blit(title, (cx - title.get_width() // 2, cy - 80))

    options = [
        ("ESC / Space / Enter", "Resume"),
        ("R",                   "Restart"),
        ("Q",                   "Quit"),
    ]
    for i, (key_hint, label) in enumerate(options):
        line = font_item.render(f"  {key_hint:<22}  {label}", True, (200, 200, 200))
        screen.blit(line, (cx - line.get_width() // 2, cy - 10 + i * 30))

    pygame.display.flip()


paused      = False
paused_surf = None
pause_clock = pygame.time.Clock()
running     = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if paused:
                    paused = False
                else:
                    paused_surf = pygame.display.get_surface().copy()
                    paused = True

            elif paused:
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    paused = False
                elif event.key == pygame.K_r:
                    match.reset(n_players=2)
                    demo_character_brains_config(match)
                    paused = False
                elif event.key == pygame.K_q:
                    running = False

    if not running:
        break

    if paused:
        screen = pygame.display.get_surface()
        screen.blit(paused_surf, (0, 0))
        draw_pause_overlay()
        pause_clock.tick(30)
        continue

    match.tick()

    if match.done:
        winner = match.winner
        label  = f"P{winner}" if winner is not None else "tie"
        print(f"Match over — {label} wins | frame {match.frame}")
        match.reset(n_players=2)
        demo_character_brains_config(match)

    renderer.render(match)

renderer.close()
