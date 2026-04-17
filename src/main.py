from distrl.utils.config import Config

def main():
    """
    Main entry point for the Distributional RL for LTM Handover framework.
    """
    print("Starting Distributional RL for LTM Handover...")
    
    # Example: Load configuration
    try:
        config = Config.get()
        print("Configuration loaded successfully.")
        print(f"UE Number: {config['simulation']['ue_number']}")
        print(f"BS Number: {config['simulation']['bs_number']}")
    except Exception as e:
        print(f"Error loading configuration: {e}")

if __name__ == "__main__":
    main()
