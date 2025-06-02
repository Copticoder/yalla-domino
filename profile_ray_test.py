import cProfile
import pstats
import io
import time
import pyspiel
from contextlib import contextmanager
import ray
from deep_cfr_ray import Orchestrator, DeepCFRActor
import psutil
import os

@contextmanager
def timer(name):
    """Context manager to time code blocks"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"\n[TIMER] {name}: {elapsed:.4f} seconds")

class ProfiledOrchestrator(Orchestrator):
    """Orchestrator with additional timing instrumentation"""
    
    def __init__(self, *args, **kwargs):
        self.timing_data = {
            'traversals': [],
            'advantage_training': [],
            'strategy_training': [],
            'network_updates': [],
            'ray_communication': []
        }
        super().__init__(*args, **kwargs)
    
    def _update_actor_networks(self):
        """Update actor networks with timing"""
        start = time.time()
        super()._update_actor_networks()
        self.timing_data['network_updates'].append(time.time() - start)
    
    def solve(self):
        """Solution logic with detailed timing"""
        import collections
        from tqdm import tqdm
        
        advantage_losses = collections.defaultdict(list)
        
        print("\n=== Starting Profiled Training ===")
        
        for i in tqdm(range(self._num_iterations), desc="Training iterations"):
            iteration_start = time.time()
            
            for p in tqdm(range(self._num_players), desc=f"Players (iteration {i+1})", leave=False):
                # Time network updates
                with timer(f"Network update (iter {i+1}, player {p})"):
                    self._update_actor_networks()
                
                # Time parallel traversals
                traversal_start = time.time()
                traversal_tasks = [actor.batch_traverse_tree_tasks.remote(p, i) 
                                 for actor in self.actors]
                
                # Time Ray communication
                ray_start = time.time()
                results = ray.get(traversal_tasks)
                ray_time = time.time() - ray_start
                self.timing_data['ray_communication'].append(ray_time)
                
                traversal_time = time.time() - traversal_start
                self.timing_data['traversals'].append(traversal_time)
                
                # Process results
                for result in results:
                    self._advantage_memories[p].add(result[0][p])
                    self._strategy_memories.add(result[1])
                
                if self._reinitialize_advantage_networks:
                    self.reinitialize_advantage_network(p)
                
                # Time advantage network training
                train_start = time.time()
                advantage_losses[p].append(self._learn_advantage_network(p))
                self.timing_data['advantage_training'].append(time.time() - train_start)
                
                print(f"Advantage loss for player {p}: {advantage_losses[p][-1]}")
            
            print(f"Iteration {i+1} time: {time.time() - iteration_start:.2f}s")
            self._iteration += 1
        
        # Time strategy network training
        strategy_start = time.time()
        policy_loss = self._learn_strategy_network()
        self.timing_data['strategy_training'].append(time.time() - strategy_start)
        
        # Print timing summary
        self._print_timing_summary()
        
        return self._policy_network, advantage_losses, policy_loss
    
    def _print_timing_summary(self):
        """Print detailed timing analysis"""
        print("\n=== Timing Summary ===")
        for key, times in self.timing_data.items():
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                print(f"{key}:")
                print(f"  Total: {total_time:.4f}s")
                print(f"  Average: {avg_time:.4f}s")
                print(f"  Count: {len(times)}")

def profile_main():
    """Main function to profile"""
    game = pyspiel.load_game('kuhn_poker')
    
    # Use smaller parameters for profiling
    solver = ProfiledOrchestrator(
        game,
        policy_network_layers=(64,),
        advantage_network_layers=(64,),
        num_iterations=10,  # Reduced for profiling
        reinitialize_advantage_networks=True,
        num_traversals=500,  # Reduced for profiling
        learning_rate=1e-3,
        batch_size_advantage=2048,
        batch_size_strategy=2048,
        memory_capacity=1e6,
        policy_network_train_steps=1000,  # Reduced for profiling
        advantage_network_train_steps=250,  # Reduced for profiling
        num_actors=4
    )
    
    print(f"Initial memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    
    start_time = time.time()
    _, advantage_losses, policy_loss = solver.solve()
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Final memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Print results
    for player, losses in list(advantage_losses.items()):
        if losses:
            print(f"Advantage for player {player}: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print("Final policy loss:", policy_loss)

def run_cprofile():
    """Run with cProfile for detailed function-level profiling"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    profile_main()
    
    profiler.disable()
    
    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions by cumulative time
    print("\n=== cProfile Results (Top 30 by cumulative time) ===")
    print(s.getvalue())
    
    # Also sort by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    ps2.print_stats(20)  # Top 20 functions by total time
    print("\n=== cProfile Results (Top 20 by total time) ===")
    print(s2.getvalue())

if __name__ == "__main__":
    print("Starting Deep CFR profiling...")
    print("=" * 60)
    
    # Run the profiled version
    run_cprofile()
    
    # Cleanup Ray
    try:
        ray.shutdown()
    except:
        pass 