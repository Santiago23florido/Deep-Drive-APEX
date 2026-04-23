from log_manager import initialize_logging
from strategy_system import StrategyManager




if __name__ == "__main__":
    # Test the strategy system
    
    
    initialize_logging(console_level="DEBUG")
    
    print("=== Strategy System Test ===\n")
    
    # Create strategy manager
    manager = StrategyManager("normal")
    
    print(f"Available strategies: {manager.list_strategies()}")
    print(f"Current strategy: {manager.get_strategy_name()}\n")
    
    # Test speed computation with different strategies
    lidar_data = np.random.rand(360)
    target_angle = 15.0
    front_distance = 1.5
    current_speed = 1.0
    
    for strategy_name in ["conservative", "normal", "aggressive"]:
        manager.set_strategy(strategy_name)
        
        speed = manager.compute_speed(lidar_data, target_angle, front_distance)
        steering = manager.compute_steering(target_angle, current_speed)
        
        print(f"{strategy_name.capitalize()} Strategy:")
        print(f"  Speed: {speed:.2f} m/s")
        print(f"  Steering: {steering:.2f}°")
        print(f"  Profile: {manager.get_profile()}\n")
        
        
