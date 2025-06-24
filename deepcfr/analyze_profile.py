import pstats
import sys

# Load the profiling data
p = pstats.Stats('results.prof')

# Sort by cumulative time and print top 20 entries
print("\nTop 20 time-consuming operations (sorted by cumulative time):")
p.sort_stats('cumulative').print_stats(20)

# Sort by internal time and print top 20 entries
print("\nTop 20 time-consuming operations (sorted by internal time):")
p.sort_stats('time').print_stats(20)

# Print callers of the most time-consuming function
print("\nCallers of the most time-consuming function:")
p.sort_stats('cumulative').print_callers(10) 