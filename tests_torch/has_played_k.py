import torch

def player_k_has_played(valid: torch.Tensor, player_id: torch.Tensor,
                player: int) -> torch.Tensor:
  """Compute a mask where player k has played in the sequence. Useful for accurately calculating the value function when it's player k's turn."""
  # assert that valid and player_id are the same shape
  assert valid.shape == player_id.shape, "valid and player_id must have the same shape"
  # assert that valid is a boolean tensor
  if valid.dtype != torch.bool:
      valid = valid.bool()
  # create a new tensor player_k_has_played with the same shape as valid 
  player_k_has_played = torch.zeros_like(valid)
  # set player_k_has_played[t] to 1 if valid[t] and player_id[t] == player
  player_k_has_played[valid & (player_id == player)] = 1
  return player_k_has_played
  
  # assert that player_id is a long tensor
def assert_tensors_equal(t1, t2, test_name=""):
    assert torch.equal(t1, t2), f"{test_name}: Expected \n{t2}\nGot \n{t1}"

def test_basic_case():
    valid = torch.tensor([True, True, True, True])
    player_id = torch.tensor([0, 1, 0, 1])
    player = 0
    expected = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    result = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result, expected, "test_basic_case")

def test_player_plays_consecutively():
    valid = torch.tensor([True, True, True, True])
    player_id = torch.tensor([0, 0, 1, 0])
    player = 0
    expected = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    result = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result, expected, "test_player_plays_consecutively")

def test_all_invalid_steps():
    valid = torch.tensor([False, False, False])
    player_id = torch.tensor([0, 0, 0])
    player = 0
    expected = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    result = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result, expected, "test_all_invalid_steps")

def test_no_steps_for_player():
    valid = torch.tensor([True, True, True])
    player_id = torch.tensor([1, 1, 1])
    player = 0
    expected = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    result = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result, expected, "test_no_steps_for_player")

def test_mixed_valid_invalid():
    valid = torch.tensor([True, False, True, False, True])
    player_id = torch.tensor([0, 1, 0, 0, 1])
    player = 0
    expected = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    result = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result, expected, "test_mixed_valid_invalid")

def test_empty_input():
    valid = torch.empty((0,), dtype=torch.bool)
    player_id = torch.empty((0,), dtype=torch.long)
    player = 0
    expected = torch.empty((0,), dtype=torch.float32)
    result = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result, expected, "test_empty_input")

    valid_b = torch.empty((0, 5), dtype=torch.bool) # With batch dim
    player_id_b = torch.empty((0, 5), dtype=torch.long)
    expected_b = torch.empty((0, 5), dtype=torch.float32)
    result_b = player_k_has_played(valid_b, player_id_b, player)
    assert_tensors_equal(result_b, expected_b, "test_empty_input_with_batch")


def test_different_valid_dtype():
    valid_int = torch.tensor([1, 0, 1], dtype=torch.int32) # 0 for False, 1 for True
    player_id = torch.tensor([0, 0, 0])
    player = 0
    expected = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    result = player_k_has_played(valid_int, player_id, player)
    assert_tensors_equal(result, expected, "test_different_valid_dtype_int")

    valid_float = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    expected_float = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    result_float = player_k_has_played(valid_float, player_id, player)
    assert_tensors_equal(result_float, expected_float, "test_different_valid_dtype_float")


def test_with_batch_dimension():
    # Time=3, Batch=2
    valid = torch.tensor([[True, True], [False, True], [True, False]], dtype=torch.bool)
    player_id = torch.tensor([[0, 1], [0, 0], [1, 0]], dtype=torch.long)
    
    player = 0
    # Expected for player 0:
    # Batch 0: [1, 0, 0] (player_id: [0,0,1]) valid: [T,F,T] -> [1,0,0]
    # Batch 1: [0, 1, 0] (player_id: [1,0,0]) valid: [T,T,F] -> [0,1,0]
    expected_p0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    result_p0 = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result_p0, expected_p0, "test_with_batch_dimension_player0")

    player = 1
    # Expected for player 1:
    # Batch 0: [0, 0, 1] (player_id: [0,0,1]) valid: [T,F,T] -> [0,0,1]
    # Batch 1: [1, 0, 0] (player_id: [1,0,0]) valid: [T,T,F] -> [1,0,0]
    expected_p1 = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    result_p1 = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result_p1, expected_p1, "test_with_batch_dimension_player1")

def test_user_example():
    valid = torch.tensor([True,True,True])
    player_id = torch.tensor([1,0,1])
    player = 1
    expected = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    result = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result, expected, "test_user_example")
    
    player = 0
    expected_p0 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    result_p0 = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result_p0, expected_p0, "test_user_example_player0")


def test_complex_sequence_batched():
    # T=5, B=2
    valid = torch.tensor([
        [True, True],    # t0
        [True, False],   # t1
        [True, True],    # t2
        [False, True],   # t3
        [True, True]     # t4
    ], dtype=torch.bool)

    player_id = torch.tensor([
        [0, 1],  # t0
        [0, 0],  # t1 (batch 1 invalid based on `valid` tensor)
        [1, 0],  # t2
        [0, 1],  # t3 (batch 0 invalid based on `valid` tensor)
        [0, 1]   # t4
    ], dtype=torch.long)

    player = 0
    # Batch 0 (col 0 of inputs): valid=[T,T,T,F,T], p_id=[0,0,1,0,0]
    #   t0: V=T, P=0. player=0. -> 1
    #   t1: V=T, P=0. player=0. -> 1
    #   t2: V=T, P=1. player=0. -> 0
    #   t3: V=F, P=0. player=0. -> 0
    #   t4: V=T, P=0. player=0. -> 1
    # Batch 0 expected: [1,1,0,0,1]

    # Batch 1 (col 1 of inputs): valid=[T,F,T,T,T], p_id=[1,0,0,1,1]
    #   t0: V=T, P=1. player=0. -> 0
    #   t1: V=F, P=0. player=0. -> 0
    #   t2: V=T, P=0. player=0. -> 1
    #   t3: V=T, P=1. player=0. -> 0
    #   t4: V=T, P=1. player=0. -> 0
    # Batch 1 expected: [0,0,1,0,0]
    
    expected_p0_corrected = torch.tensor([
        [1.0, 0.0], #t0
        [1.0, 0.0], #t1
        [0.0, 1.0], #t2
        [0.0, 0.0], #t3
        [1.0, 0.0]  #t4
    ], dtype=torch.float32)

    result_p0 = player_k_has_played(valid, player_id, player)
    assert_tensors_equal(result_p0, expected_p0_corrected, "test_complex_sequence_batched_player0")


if __name__ == "__main__":
    test_basic_case()
    test_player_plays_consecutively()
    test_all_invalid_steps()
    test_no_steps_for_player()
    test_mixed_valid_invalid()
    test_empty_input()
    test_different_valid_dtype()
    test_with_batch_dimension()
    test_user_example()
    test_complex_sequence_batched()
    print("All tests passed (if no assertion errors)!")