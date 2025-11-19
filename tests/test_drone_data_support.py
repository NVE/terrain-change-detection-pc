"""
Tests for drone scanning data support using simplified structure.

Drone data structure: area/time_period/*.laz (no 'data' subdirectory)
Hoydedata structure: area/time_period/data/*.laz
"""

import pytest
from pathlib import Path
from terrain_change_detection.preprocessing import DataDiscovery, BatchLoader


class TestDroneDataDiscovery:
    """Test DataDiscovery with drone data source."""
    
    @pytest.fixture
    def drone_data_dir(self):
        """Path to drone scanning data."""
        return "data/drone_scanning_data"
    
    def test_scan_drone_areas(self, drone_data_dir):
        """Test scanning drone data with source_type='drone'."""
        discovery = DataDiscovery(drone_data_dir, source_type='drone')
        areas = discovery.scan_areas()
        
        # Should find areas if data exists
        if not Path(drone_data_dir).exists():
            pytest.skip("Drone scanning data not available")
        
        assert isinstance(areas, dict)
        
        # If areas found, validate structure
        for area_name, area_info in areas.items():
            assert area_info.area_name == area_name
            assert len(area_info.datasets) > 0
            
            for time_period, dataset in area_info.datasets.items():
                assert dataset.area_name == area_name
                assert dataset.time_period == time_period
                assert len(dataset.laz_files) > 0
                
                # Verify files exist
                for file_path in dataset.laz_files:
                    assert file_path.exists()
                    assert file_path.suffix.lower() in ['.las', '.laz']
    
    def test_drone_vs_hoydedata_structure(self):
        """Test that source_type correctly handles different structures."""
        drone_dir = "data/drone_scanning_data"
        hoydedata_dir = "data/raw"
        
        # Drone: no 'data' subdirectory
        if Path(drone_dir).exists():
            discovery_drone = DataDiscovery(drone_dir, source_type='drone')
            assert discovery_drone.data_dir_name is None
            assert discovery_drone.metadata_dir_name is None
        
        # Hoydedata: requires 'data' subdirectory
        if Path(hoydedata_dir).exists():
            discovery_hoydedata = DataDiscovery(hoydedata_dir, source_type='hoydedata')
            assert discovery_hoydedata.data_dir_name == 'data'
            assert discovery_hoydedata.metadata_dir_name == 'metadata'
    
    def test_load_drone_dataset(self, drone_data_dir):
        """Test loading drone dataset with BatchLoader."""
        if not Path(drone_data_dir).exists():
            pytest.skip("Drone scanning data not available")
        
        discovery = DataDiscovery(drone_data_dir, source_type='drone')
        areas = discovery.scan_areas()
        
        if not areas:
            pytest.skip("No drone data found")
        
        # Get first dataset
        first_area = next(iter(areas.values()))
        first_dataset = next(iter(first_area.datasets.values()))
        
        # Load with BatchLoader
        batch_loader = BatchLoader(streaming_mode=False)
        result = batch_loader.load_dataset(first_dataset)
        
        assert 'points' in result
        assert 'metadata' in result
        assert len(result['points']) > 0
        assert result['points'].shape[1] == 3  # X, Y, Z
    
    def test_streaming_mode_drone(self, drone_data_dir):
        """Test streaming mode with drone data."""
        if not Path(drone_data_dir).exists():
            pytest.skip("Drone scanning data not available")
        
        discovery = DataDiscovery(drone_data_dir, source_type='drone')
        areas = discovery.scan_areas()
        
        if not areas:
            pytest.skip("No drone data found")
        
        # Get first dataset
        first_area = next(iter(areas.values()))
        first_dataset = next(iter(first_area.datasets.values()))
        
        # Prepare for streaming
        batch_loader = BatchLoader(streaming_mode=True)
        result = batch_loader.load_dataset(first_dataset)
        
        assert result['mode'] == 'streaming'
        assert 'file_paths' in result
        assert len(result['file_paths']) > 0


class TestHoydedataCompatibility:
    """Ensure hoydedata.no still works with source_type parameter."""
    
    def test_hoydedata_discovery(self):
        """Test that hoydedata discovery still works."""
        hoydedata_dir = "data/raw"
        
        if not Path(hoydedata_dir).exists():
            pytest.skip("Hoydedata not available")
        
        discovery = DataDiscovery(hoydedata_dir, source_type='hoydedata')
        areas = discovery.scan_areas()
        
        assert isinstance(areas, dict)
    
    def test_backward_compatibility_default(self):
        """Test that default behavior is backward compatible (hoydedata)."""
        # When source_type is not specified, should default to hoydedata behavior
        discovery = DataDiscovery("data/raw")
        
        # Should still have data_dir_name set
        assert discovery.data_dir_name is not None
