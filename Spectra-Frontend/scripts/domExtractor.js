// scripts/domExtractor.js

const domExtractor = {
    // Builds an ordered array of text blocks and images from top to bottom
    buildReadingQueue: function() {
        console.log("🔍 Scanning webpage for readable content...");
        const queue = [];
        
        // Prioritize the main content area to avoid reading menus and footers
        const mainNode = document.querySelector('main, article, [role="main"]') || document.body;

        // Select text blocks and images in their natural DOM sequence
        const elements = mainNode.querySelectorAll('h1, h2, h3, h4, h5, p, li, img');

        elements.forEach(el => {
            // Ignore hidden elements or junk inside navbars/footers
            if (el.closest('nav, footer, header, aside, .sidebar, .menu, .ad, [aria-hidden="true"]')) {
                return;
            }

            const rect = el.getBoundingClientRect();
            // Skip elements that are invisible on the screen
            if (rect.width === 0 || rect.height === 0) {
                return;
            }

            if (el.tagName === 'IMG') {
                // Ensure it's a real image and not a tiny decorative icon
                if (rect.width > 50 && rect.height > 50 && el.src && el.src.startsWith('http')) {
                    console.log(`🖼️ Found Image: ${el.src.substring(0, 50)}...`);
                    queue.push({ type: 'image', url: el.src });
                }
            } else {
                const text = el.innerText.trim();
                // Ignore stray characters or empty tags
                if (text.length > 5) {
                    queue.push({ type: 'text', content: text });
                }
            }
        });
        
        console.log(`📊 Scan Complete: Found a total of ${queue.length} readable items (text + images).`);
        return queue;
    }
};