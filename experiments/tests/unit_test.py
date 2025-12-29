import unittest
import torch
from torch import nn
from models.som_layer import SOMLayer

class TestSOMLayer(unittest.TestCase):
    def setUp(self):
        self.som_layer = SOMLayer(map_size=(8, 8), latent_dim=10, Tmax=1.0, Tmin=0.1, lr_max=0.5, lr_min=0.01, total_epochs=100)
        self.prototypes = torch.rand((64, 10))  # Assuming an 8x8 SOM grid and 10-dimensional feature space

    def test_index_to_position(self):
        """Test conversion from index to 2D SOM grid coordinates."""
        index = 10
        expected_position = (1, 2)
        calculated_position = self.som_layer.index_to_position(torch.tensor([index]))
        self.assertEqual(tuple(calculated_position.numpy()[0]), expected_position)

    def test_update_prototypes(self):
        """Test prototypes are updated correctly given dummy inputs and BMUs."""
        initial_prototypes = self.prototypes.clone()
        x = torch.rand((5, 10))
        bmu_indices = torch.tensor([0, 1, 2, 3, 4])
        self.som_layer.prototypes = nn.Parameter(initial_prototypes)
        self.som_layer.update_prototypes(x, bmu_indices)
        
        self.assertFalse(torch.equal(self.som_layer.prototypes.data, initial_prototypes))

if __name__ == '__main__':
    unittest.main()
