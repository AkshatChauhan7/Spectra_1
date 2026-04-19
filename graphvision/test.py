import json

from graphvision_ai import GraphExtractor 

def main():
    print("🚀 Initializing GraphVision AI...")
    
    # 1. Boot up the models
    extractor = GraphExtractor()
    
    # 2. Point it to your test image
    # Update this path to where your actual test image is located!
    test_image = "2.png" 
    
    print(f"\n📸 Extracting data from: {test_image}")
    
    # 3. Run the extraction (setting show=True so you can see the bounding boxes)
    results = extractor.extract_data(test_image, show=True)
    
    # 4. Print the beautiful structured output!
    print("\n✅ Final Output:")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()