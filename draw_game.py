"""Draw Dominoes Implemented in Python"""

import pyspiel

from open_spiel.python.games.block_dominoes import BlockDominoesState, Action, BlockDominoesObserver, _GAME_INFO, _NUM_PLAYERS, _ACTIONS_STR


class DrawDominoesGame(pyspiel.Game):
  """A Python version of Block Dominoes."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return DrawDominoesState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return BlockDominoesObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params
    )
    
class DrawDominoesState(BlockDominoesState):
  def __init__(self, game):
    super().__init__(game)

  def get_legal_actions(self, player):
    """Returns a list of legal actions."""
    assert player >= 0

    actions = []
    hand = self.hands[player]

    # first move, no open edges
    if not self.open_edges:
      for tile in hand:
        actions.append(Action(player, tile, None))
    else:
      for tile in hand:
        if tile[0] in self.open_edges:
          actions.append(Action(player, tile, tile[0]))
        if tile[0] != tile[1] and tile[1] in self.open_edges:
          actions.append(Action(player, tile, tile[1]))
          
    # draw from the deck until we have a legal action or the deck is empty
    while len(self.deck) > 0 and len(actions) == 0:
        tile = self.deck.pop(0)
        # add it to player's hand 
        self.hands[player].append(tile)
        condition_1, condition_2 = tile[0] in self.open_edges, tile[0] != tile[1] and tile[1] in self.open_edges
        if condition_1 or condition_2:
            actions.append(Action(player, tile, tile[0] if condition_1 else tile[1]))
            break
    actions_idx = [_ACTIONS_STR.index(str(action)) for action in actions]
    actions_idx.sort()
    return actions_idx

_GAME_TYPE = pyspiel.GameType(
    short_name="python_draw_dominoes",
    long_name="python_draw_dominoes",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)

pyspiel.register_game(_GAME_TYPE, DrawDominoesGame)
# test the new Draw Dominoes State

game = pyspiel.load_game('python_draw_dominoes')
state = game.new_initial_state()

while not state.is_terminal():
    state.apply_action(state.legal_actions()[0])
    print(state)