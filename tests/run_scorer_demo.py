from visibility import Scorer

s = Scorer("tiny")

print("Candidate count:", s.n_candidates)
print("Coverage of first camera:", s.score([0]))
print("Coverage of two cameras:", s.score([0, 10]))
print("Coverage grid shape:", s.covered_grid([0]).shape)