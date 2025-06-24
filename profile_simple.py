"""
Simple profiling script for Deep CFR - no additional dependencies required.
This will help identify performance bottlenecks in your code.
"""

import time
import pyspiel
from deepcfr.deep_cfr_ray import Orchestrator
import ray

class TimingProfiler:
    """Simple timing profiler that tracks execution times"""
    
    def __init__(self):
        self.timings = {}
    
    def start(self, name):
        """Start timing a section"""
        if name not in self.timings:
            self.timings[name] = {'times': [], 'start': None}
        self.timings[name]['start'] = time.time()
    
    def end(self, name):
        """End timing a section"""
        if name in self.timings and self.timings[name]['start'] is not None:
            elapsed = time.time() - self.timings[name]['start']
            self.timings[name]['times'].append(elapsed)
            self.timings[name]['start'] = None
    
    def report(self):
        """Print timing report"""
        print("\n" + "="*80)
        print("TIMING REPORT")
        print("="*80)
        
        total_time = sum(sum(data['times']) for data in self.timings.values() if data['times'])
        
        # Sort by total time
        sorted_timings = sorted(
            [(name, data) for name, data in self.timings.items() if data['times']],
            key=lambda x: sum(x[1]['times']),
            reverse=True
        )
        
        for name, data in sorted_timings:
            times = data['times']
            total = sum(times)
            avg = total / len(times) if times else 0
            percentage = (total / total_time * 100) if total_time > 0 else 0
            
            print(f"\n{name}:")
            print(f"  Total time: {total:.4f}s ({percentage:.1f}%)")
            print(f"  Calls: {len(times)}")
            print(f"  Average: {avg:.4f}s")
            if len(times) > 1:
                print(f"  Min/Max: {min(times):.4f}s / {max(times):.4f}s")


# Global profiler instance
profiler = TimingProfiler()


class ProfiledOrchestrator(Orchestrator):
    """Orchestrator with profiling instrumentation"""
    
    def solve(self):
        """Instrumented solve method"""
        import collections
        from tqdm import tqdm
        
        advantage_losses = collections.defaultdict(list)
        
        print("\n=== Starting Profiled Training ===")
        overall_start = time.time()
        
        for i in tqdm(range(self._num_iterations), desc="Training iterations"):
            iteration_start = time.time()
            
            for p in tqdm(range(self._num_players), desc=f"Players (iteration {i+1})", leave=False):
                # Profile network updates
                profiler.start('network_updates')
                self._update_actor_networks()
                profiler.end('network_updates')
                
                # Profile traversals setup
                profiler.start('traversal_setup')
                traversal_tasks = [actor.batch_traverse_tree_tasks.remote(p, i) 
                                 for actor in self.actors]
                profiler.end('traversal_setup')
                
                # Profile Ray communication
                profiler.start('ray_get')
                results = ray.get(traversal_tasks)
                profiler.end('ray_get')
                
                # Profile data processing
                profiler.start('data_processing')
                for result in results:
                    self._advantage_memories[p].add(result[0][p])
                    self._strategy_memories.add(result[1])
                profiler.end('data_processing')
                
                # Profile network reinitialization if applicable
                if self._reinitialize_advantage_networks:
                    profiler.start('reinitialize_network')
                    self.reinitialize_advantage_network(p)
                    profiler.end('reinitialize_network')
                
                # Profile advantage network training
                profiler.start('advantage_training')
                advantage_losses[p].append(self._learn_advantage_network(p))
                profiler.end('advantage_training')
                
                print(f"Player {p} advantage loss: {advantage_losses[p][-1]}")
            
            iteration_time = time.time() - iteration_start
            print(f"Iteration {i+1} completed in {iteration_time:.2f}s")
            self._iteration += 1
        
        # Profile strategy network training
        profiler.start('strategy_training')
        policy_loss = self._learn_strategy_network()
        profiler.end('strategy_training')
        
        overall_time = time.time() - overall_start
        print(f"\nTotal training time: {overall_time:.2f}s")
        
        return self._policy_network, advantage_losses, policy_loss
    
    def _learn_advantage_network(self, player):
        """Instrumented advantage network learning"""
        profiler.start('advantage_learn_total')
        
        for step in range(self._advantage_network_train_steps):
            # Profile data sampling
            profiler.start('advantage_sampling')
            if self._batch_size_advantage:
                memory_size = len(self._advantage_memories[player])
                if self._batch_size_advantage > memory_size:
                    profiler.end('advantage_sampling')
                    profiler.end('advantage_learn_total')
                    return None
                samples = self._advantage_memories[player].sample(self._batch_size_advantage)
            else:
                memory_size = len(self._advantage_memories[player])
                if memory_size == 0:
                    profiler.end('advantage_sampling')
                    profiler.end('advantage_learn_total')
                    return None
                samples = self._advantage_memories[player].sample(memory_size)
            profiler.end('advantage_sampling')
            
            if not samples:
                profiler.end('advantage_learn_total')
                return None
            
            # Profile data preparation
            profiler.start('advantage_data_prep')
            import torch
            import numpy as np
            
            info_states = []
            advantages = []
            iterations = []
            for s in samples:
                info_states.append(s.info_state)
                advantages.append(s.advantage)
                iterations.append([s.iteration])
            
            advantages = torch.FloatTensor(np.array(advantages))
            iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
            states_tensor = torch.FloatTensor(np.array(info_states))
            profiler.end('advantage_data_prep')
            
            # Profile network forward/backward pass
            profiler.start('advantage_network_pass')
            self._optimizer_advantages[player].zero_grad()
            outputs = self._advantage_networks[player](states_tensor)
            loss_advantages = self._loss_advantages(iters * outputs, iters * advantages)
            loss_advantages.backward()
            self._optimizer_advantages[player].step()
            profiler.end('advantage_network_pass')
        
        profiler.end('advantage_learn_total')
        return loss_advantages.detach().numpy()


def run_profiling():
    """Run the profiling analysis"""
    game = pyspiel.load_game('kuhn_poker')
    
    print("Setting up Deep CFR with profiling...")
    print("Parameters:")
    print("  - Iterations: 10")
    print("  - Traversals: 500") 
    print("  - Actors: 4")
    print("  - Batch sizes: 2048")
    print()
    
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
        policy_network_train_steps=1000,  # Reduced
        advantage_network_train_steps=250,  # Reduced
        num_actors=4
    )
    
    start_time = time.time()
    _, advantage_losses, policy_loss = solver.solve()
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*80}")
    print(f"PROFILING COMPLETE")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # Show the timing breakdown
    profiler.report()
    
    # Analyze bottlenecks
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Calculate percentages
    timings_summary = {}
    for name, data in profiler.timings.items():
        if data['times']:
            timings_summary[name] = sum(data['times'])
    
    total_tracked = sum(timings_summary.values())
    untracked = total_time - total_tracked
    
    print(f"\nTotal time tracked: {total_tracked:.2f}s ({total_tracked/total_time*100:.1f}%)")
    print(f"Untracked time: {untracked:.2f}s ({untracked/total_time*100:.1f}%)")
    
    # Identify main bottlenecks
    sorted_components = sorted(timings_summary.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop bottlenecks:")
    for i, (component, time_spent) in enumerate(sorted_components[:5]):
        percentage = time_spent / total_time * 100
        print(f"{i+1}. {component}: {time_spent:.2f}s ({percentage:.1f}%)")
    
    # Cleanup
    try:
        ray.shutdown()
    except:
        pass


if __name__ == "__main__":
    print("Deep CFR Performance Profiling")
    print("="*80)
    run_profiling() 