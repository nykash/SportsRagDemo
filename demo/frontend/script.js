// Video background functionality
const backgroundVideo = document.getElementById('backgroundVideo');
// Ensure video doesn't loop
backgroundVideo.loop = false;

// Check if browser supports .mov format
function checkVideoSupport() {
    const video = document.createElement('video');
    const canPlayMov = video.canPlayType('video/quicktime') || video.canPlayType('video/mp4');
    console.log('Browser video support check:');
    console.log('- QuickTime (.mov):', video.canPlayType('video/quicktime'));
    console.log('- MP4:', video.canPlayType('video/mp4'));
    console.log('- WebM:', video.canPlayType('video/webm'));
    return canPlayMov;
}

let videoClips = [
    'assets/visualization_1_128.64_140.48_1.mp4',
    'assets/visualization_1_37.44_43.2_1.mp4',
    'assets/visualization_1_5.76_16.96_1.mp4',
    'assets/visualization_1_78.4_85.76_1.mp4',
    'assets/visualization_1_99.2_109.12_1.mp4'
];

let errorCount = 0;
const MAX_ERRORS = 10;
let isPlaying = false;
let isSwitching = false; // Flag to prevent premature switching
let videoStartTime = 0; // Track when video started playing
let shuffledPlaylist = [];
let currentPlaylistIndex = -1;
let lastPlayedVideo = null; // Track last played video to avoid immediate repeats

// Shuffle array function (Fisher-Yates)
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// Initialize shuffled playlist
function initializePlaylist() {
    shuffledPlaylist = shuffleArray(videoClips);
    
    // If the first video in the new playlist is the same as the last played, reshuffle
    if (lastPlayedVideo && shuffledPlaylist[0] === lastPlayedVideo && shuffledPlaylist.length > 1) {
        console.log('First video in new playlist matches last played, reshuffling...');
        // Swap first video with a random other one
        const swapIndex = Math.floor(Math.random() * (shuffledPlaylist.length - 1)) + 1;
        [shuffledPlaylist[0], shuffledPlaylist[swapIndex]] = [shuffledPlaylist[swapIndex], shuffledPlaylist[0]];
    }
    
    currentPlaylistIndex = -1;
    console.log('Playlist initialized:', shuffledPlaylist.map(v => v.split('/').pop()));
}

// Get next video from shuffled playlist
function getNextVideo() {
    // If we've played all videos, reshuffle
    if (currentPlaylistIndex >= shuffledPlaylist.length - 1) {
        console.log('All videos played, reshuffling playlist');
        initializePlaylist();
        currentPlaylistIndex = 0;
    } else {
        currentPlaylistIndex++;
    }
    
    let videoPath = shuffledPlaylist[currentPlaylistIndex];
    
    // Double-check: if somehow we got the same video, skip to next
    if (videoPath === lastPlayedVideo && shuffledPlaylist.length > 1) {
        console.warn('Same video selected, skipping to next');
        if (currentPlaylistIndex < shuffledPlaylist.length - 1) {
            currentPlaylistIndex++;
            videoPath = shuffledPlaylist[currentPlaylistIndex];
        } else {
            // We're at the end, reshuffle
            initializePlaylist();
            currentPlaylistIndex = 0;
            videoPath = shuffledPlaylist[currentPlaylistIndex];
        }
    }
    
    console.log('Next video:', {
        index: currentPlaylistIndex,
        totalVideos: shuffledPlaylist.length,
        path: videoPath.split('/').pop(),
        lastPlayed: lastPlayedVideo ? lastPlayedVideo.split('/').pop() : 'none',
        remaining: shuffledPlaylist.length - currentPlaylistIndex - 1,
        fullPlaylist: shuffledPlaylist.map(v => v.split('/').pop())
    });
    
    return videoPath;
}

// Function to play the next video from playlist
function playRandomVideo() {
    if (errorCount >= MAX_ERRORS) {
        console.error('Too many video errors. Stopping video playback.');
        return;
    }

    if (isPlaying || isSwitching) {
        console.log('Video already playing or switching, skipping new play request');
        return;
    }

    isSwitching = true; // Mark that we're intentionally switching
    const previousPath = backgroundVideo.src;
    const videoPath = getNextVideo();
    
    // Update last played video
    lastPlayedVideo = videoPath;
    
    console.log('Switching video:', {
        previous: previousPath ? previousPath.split('/').pop() : 'none',
        current: videoPath.split('/').pop()
    });
    
    isPlaying = true;
    
    // Pause current video
    backgroundVideo.pause();
    
    // Set the new source directly (browser will handle the transition)
    const fullPath = new URL(videoPath, window.location.href).href;
    backgroundVideo.src = fullPath;
    
    // Load the new video
    backgroundVideo.load();
        
    // Verify the source was set correctly
    console.log('Video source set to:', backgroundVideo.src);
    
    // Wait for video to be ready
    const tryPlay = () => {
            if (backgroundVideo.readyState >= 2) { // HAVE_CURRENT_DATA
                const playPromise = backgroundVideo.play();
                
                if (playPromise !== undefined) {
                    playPromise
                        .then(() => {
                            // Ensure video doesn't loop
                            backgroundVideo.loop = false;
                            console.log('Video playing successfully:', videoPath);
                            console.log('Video duration:', backgroundVideo.duration, 'seconds');
                            console.log('Video loop:', backgroundVideo.loop);
                            videoStartTime = Date.now(); // Record when video started
                            errorCount = 0; // Reset error count on success
                            isPlaying = false;
                            isSwitching = false; // Done switching, allow natural ending
                        })
                        .catch(error => {
                            console.error('Error playing video:', error);
                            console.error('Video error details:', {
                                code: backgroundVideo.error?.code,
                                message: backgroundVideo.error?.message,
                                path: videoPath,
                                readyState: backgroundVideo.readyState
                            });
                            errorCount++;
                            isPlaying = false;
                            isSwitching = false;
                            // Try next video after a short delay
                            setTimeout(() => {
                                if (errorCount < MAX_ERRORS) {
                                    playRandomVideo();
                                }
                            }, 1000);
                        });
                } else {
                    isPlaying = false;
                    isSwitching = false;
                }
            } else {
                // Wait a bit more for video to load
                setTimeout(tryPlay, 50);
            }
        };
        
        // Start trying to play
        if (backgroundVideo.readyState >= 2) {
            tryPlay();
        } else {
            const loadHandler = () => {
                tryPlay();
            };
            backgroundVideo.addEventListener('loadeddata', loadHandler, { once: true });
            // Fallback timeout
            setTimeout(() => {
                if (isSwitching) {
                    console.warn('Video took too long to load, trying to play anyway');
                    backgroundVideo.removeEventListener('loadeddata', loadHandler);
                    tryPlay();
                }
            }, 2000);
        }
}

// Track video progress
let lastLoggedTime = 0;
backgroundVideo.addEventListener('timeupdate', () => {
    const currentTime = backgroundVideo.currentTime;
    const duration = backgroundVideo.duration;
    
    // Log progress every 2 seconds
    if (Math.floor(currentTime) !== Math.floor(lastLoggedTime) && Math.floor(currentTime) % 2 === 0) {
        console.log(`Video progress: ${currentTime.toFixed(1)}s / ${duration.toFixed(1)}s (${((currentTime/duration)*100).toFixed(1)}%)`);
    }
    lastLoggedTime = currentTime;
    
    // Check if we're near the end (within 0.1 seconds)
    if (duration > 0 && Math.abs(currentTime - duration) < 0.1 && !isSwitching) {
        console.log('Video near end, waiting for ended event...');
    }
});

// When video ends, play another random one
backgroundVideo.addEventListener('ended', () => {
    console.log('=== VIDEO ENDED EVENT FIRED ===');
    const timePlayed = videoStartTime > 0 ? Date.now() - videoStartTime : 0;
    
    console.log('Ended event details:', {
        isSwitching,
        duration: backgroundVideo.duration,
        currentTime: backgroundVideo.currentTime,
        timePlayed: timePlayed,
        videoSrc: backgroundVideo.src ? backgroundVideo.src.split('/').pop() : 'none'
    });
    
    // Only check if we're not currently switching videos
    if (!isSwitching && backgroundVideo.duration > 0) {
        console.log('✓ Video ended naturally, playing next video');
        console.log('  Video finished at:', backgroundVideo.currentTime.toFixed(2), 'of', backgroundVideo.duration.toFixed(2));
        if (videoStartTime > 0) {
            console.log('  Time played:', (timePlayed / 1000).toFixed(2), 'seconds');
        }
        isPlaying = false;
        playRandomVideo();
    } else {
        console.log('✗ Ignoring ended event - isSwitching:', isSwitching, 'duration:', backgroundVideo.duration);
    }
});

// Handle video loading events
backgroundVideo.addEventListener('loadeddata', () => {
    console.log('Video data loaded');
});

backgroundVideo.addEventListener('canplay', () => {
    console.log('Video can play');
});

backgroundVideo.addEventListener('canplaythrough', () => {
    console.log('Video can play through');
});

// Handle video errors
backgroundVideo.addEventListener('error', (e) => {
    const error = backgroundVideo.error;
    if (error) {
        console.error('Video error code:', error.code);
        console.error('Video error message:', error.message);
        console.error('Current video path:', backgroundVideo.src);
        
        let errorMessage = 'Unknown error';
        switch(error.code) {
            case 1: errorMessage = 'MEDIA_ERR_ABORTED - Video loading aborted'; break;
            case 2: errorMessage = 'MEDIA_ERR_NETWORK - Network error while loading video'; break;
            case 3: errorMessage = 'MEDIA_ERR_DECODE - Video decoding error'; break;
            case 4: errorMessage = 'MEDIA_ERR_SRC_NOT_SUPPORTED - Video format not supported'; break;
        }
        console.error('Error type:', errorMessage);
    }
    
    errorCount++;
    // Try playing another video if one fails
    if (errorCount < MAX_ERRORS) {
        setTimeout(() => playRandomVideo(), 1000);
    }
});

// Handle stalled video
backgroundVideo.addEventListener('stalled', () => {
    console.warn('Video stalled, trying to recover');
});

// Start playing a random video when page loads
window.addEventListener('load', () => {
    console.log('Page loaded, starting video playback');
    console.log('User agent:', navigator.userAgent);
    
    // Test backend connection
    testBackendConnection().then(isConnected => {
        if (!isConnected) {
            console.warn('Backend not reachable. Some features may not work.');
        }
    });
    
    // Check video support
    const supportsVideo = checkVideoSupport();
    if (!supportsVideo) {
        console.warn('Browser may not support video format.');
    }
    
    // Initialize shuffled playlist
    initializePlaylist();
    
    // Small delay to ensure video element is ready
    setTimeout(() => {
        playRandomVideo();
    }, 100);
});

// Also try when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log('DOM loaded, initializing video');
    });
} else {
    console.log('DOM already ready');
}

// Search functionality
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const suggestions = document.getElementById('suggestions');

// Backend API configuration
const API_BASE_URL = 'http://127.0.0.1:5000';  // Flask backend URL

// Get selected tags from checkboxes
function getSelectedTags() {
    const checkboxes = document.querySelectorAll('#tagsMultiselect input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// No longer need S3 config - videos are served directly from backend

// Videos are now served directly from backend, no S3 client needed

// Test backend connection
async function testBackendConnection() {
    try {
        // Try to access the health check endpoint
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            mode: 'cors'
        });
        if (response.ok) {
            const data = await response.json();
            console.log('✓ Backend connection successful:', data);
            return true;
        } else {
            console.warn('Backend responded but with error:', response.status);
            return false;
        }
    } catch (error) {
        console.error('✗ Backend connection test failed:', error);
        console.error('Make sure the backend is running on', API_BASE_URL);
        return false;
    }
}


// Handle search submission
function handleSearch() {
    const query = searchInput.value.trim();
    if (query) {
        console.log('Searching for:', query);
        // Add loading state
        searchButton.classList.add('loading');
        searchButton.disabled = true;
        
        // Call backend retrieve API
        // Try different possible routes
        const possibleRoutes = ['/retrieve', '/api/retrieve', '/retrieve/'];
        let apiUrl = `${API_BASE_URL}${possibleRoutes[0]}`;
        console.log('Calling backend API:', apiUrl);
        console.log('Request payload:', { question: query, tags: [] });
        
        // Get selected tags
        const selectedTags = getSelectedTags();
        
        fetch(apiUrl, {
            method: 'POST',
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: query,
                tags: selectedTags
            })
        })
        .then(response => {
            console.log('Response status:', response.status, response.statusText);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Search results:', data);
            searchButton.classList.remove('loading');
            searchButton.disabled = false;
            
            if (data.results && data.results.length > 0) {
                // Sort results by score (highest first)
                const sortedResults = [...data.results].sort((a, b) => (b.score || 0) - (a.score || 0));
                
                // Update video clips with search results
                updateVideoClips(sortedResults);
                // Display all results
                displayResults(sortedResults, data.count);
                // Don't play background video - each card will handle its own video
            } else {
                console.warn('No results found');
                hideResults();
                alert('No results found for your search.');
            }
        })
        .catch(error => {
            console.error('Search error:', error);
            console.error('Error details:', {
                message: error.message,
                stack: error.stack,
                apiUrl: apiUrl
            });
            searchButton.classList.remove('loading');
            searchButton.disabled = false;
            
            let errorMessage = 'Error searching. ';
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                errorMessage += `Cannot connect to backend at ${API_BASE_URL}. Please make sure the server is running on 127.0.0.1:5000.`;
            } else {
                errorMessage += `Error: ${error.message}`;
            }
            alert(errorMessage);
        });
    }
}

// Update video clips array with search results
function updateVideoClips(results) {
    // Videos are now served directly from backend
    videoClips = results.map(result => {
        // URL is already provided by backend
        if (result.url) {
            // Make sure it's a full URL if it's relative
            return result.url.startsWith('http') ? result.url : `${API_BASE_URL}${result.url}`;
        }
        return '';
    }).filter(url => url !== '');
    // Reset playlist to use new results
    initializePlaylist();
    console.log('Updated video clips with search results:', videoClips.length);
}

// Play video from a URL
function playVideoFromUrl(url) {
    if (isPlaying || isSwitching) {
        console.log('Video already playing or switching, skipping new play request');
        return;
    }

    isSwitching = true;
    isPlaying = true;
    
    // Pause current video
    backgroundVideo.pause();
    
    // Set the new source
    backgroundVideo.src = url;
    
    // Load the new video
    backgroundVideo.load();
    
    console.log('Loading video from URL:', url);
    
    // Wait for video to be ready
    const tryPlay = () => {
        if (backgroundVideo.readyState >= 2) {
            const playPromise = backgroundVideo.play();
            
            if (playPromise !== undefined) {
                playPromise
                    .then(() => {
                        backgroundVideo.loop = false;
                        console.log('Video playing successfully from URL');
                        videoStartTime = Date.now();
                        errorCount = 0;
                        isPlaying = false;
                        isSwitching = false;
                    })
                    .catch(error => {
                        console.error('Error playing video:', error);
                        errorCount++;
                        isPlaying = false;
                        isSwitching = false;
                    });
            } else {
                isPlaying = false;
                isSwitching = false;
            }
        } else {
            setTimeout(tryPlay, 50);
        }
    };
    
    if (backgroundVideo.readyState >= 2) {
        tryPlay();
    } else {
        const loadHandler = () => {
            tryPlay();
        };
        backgroundVideo.addEventListener('loadeddata', loadHandler, { once: true });
        setTimeout(() => {
            if (isSwitching) {
                console.warn('Video took too long to load');
                backgroundVideo.removeEventListener('loadeddata', loadHandler);
                tryPlay();
            }
        }, 2000);
    }
}

// Handle Enter key press
searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        handleSearch();
    }
});

// Handle search button click
searchButton.addEventListener('click', handleSearch);

// Handle input changes for suggestions (optional)
let debounceTimer;
searchInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    
    clearTimeout(debounceTimer);
    
    if (query.length > 0) {
        debounceTimer = setTimeout(() => {
            // TODO: Implement suggestion API call
            // fetch(`/api/suggestions?q=${encodeURIComponent(query)}`)
            //     .then(response => response.json())
            //     .then(data => displaySuggestions(data))
            //     .catch(error => console.error('Suggestions error:', error));
        }, 300);
    } else {
        hideSuggestions();
    }
});

// Display suggestions
function displaySuggestions(suggestionList) {
    if (!suggestionList || suggestionList.length === 0) {
        hideSuggestions();
        return;
    }
    
    suggestions.innerHTML = '';
    suggestionList.forEach(item => {
        const suggestionItem = document.createElement('div');
        suggestionItem.className = 'suggestion-item';
        suggestionItem.textContent = item;
        suggestionItem.addEventListener('click', () => {
            searchInput.value = item;
            hideSuggestions();
            handleSearch();
        });
        suggestions.appendChild(suggestionItem);
    });
    
    suggestions.classList.add('show');
}

// Hide suggestions
function hideSuggestions() {
    suggestions.classList.remove('show');
    suggestions.innerHTML = '';
}

// Hide suggestions when clicking outside
document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !suggestions.contains(e.target) && !searchButton.contains(e.target)) {
        hideSuggestions();
    }
});

// Focus search input after video starts
backgroundVideo.addEventListener('loadeddata', () => {
    searchInput.focus();
});

// Results display functionality
const resultsContainer = document.getElementById('resultsContainer');
const resultsGrid = document.getElementById('resultsGrid');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');

// Display search results
function displayResults(results, count) {
    // Mark page as having results
    document.querySelector('.page-wrapper').classList.add('has-results');
    
    // Add scrolled class to search header for smooth transition
    const searchHeader = document.querySelector('.search-header');
    searchHeader.classList.add('scrolled');
    
    // Stop and hide background video after a short delay to allow transition
    setTimeout(() => {
        backgroundVideo.pause();
        backgroundVideo.style.display = 'none';
        
        // Change background to white
        document.body.style.background = '#ffffff';
    }, 300);
    
    // Show results container
    resultsContainer.style.display = 'block';
    
    // Update count
    resultsCount.textContent = `${count} result${count !== 1 ? 's' : ''} found`;
    
    // Clear previous results
    resultsGrid.innerHTML = '';
    
    // Create a card for each result
    results.forEach((result, index) => {
        const card = createResultCard(result, index);
        resultsGrid.appendChild(card);
    });
    
    // Smooth scroll to results after transition
    setTimeout(() => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 100);
}

// Create a result card element
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    // Format time range
    const timeRange = `${formatTime(result.time_start)} - ${formatTime(result.time_stop)}`;
    
    // Format events - display action and jersey number
    let eventsHtml = '';
    if (result.events && Array.isArray(result.events) && result.events.length > 0) {
        // Events is an array of action objects
        const actionTags = result.events
            .filter(action => action && typeof action === 'object')
            .map(action => {
                const actionName = action.action || action.event_type || 'action';
                const jerseyNumber = action.jersey_number;
                
                // Format as "action #jersey" or just "action" if no jersey
                if (jerseyNumber) {
                    return `${actionName} #${jerseyNumber}`;
                } else {
                    return actionName;
                }
            })
            .filter(tag => tag); // Remove empty strings
        
        if (actionTags.length > 0) {
            eventsHtml = `
                <div class="result-events">
                    <div class="result-events-title">Actions</div>
                    <div class="result-events-list">
                        ${actionTags.map(tag => `<span class="result-event-tag">${escapeHtml(tag)}</span>`).join('')}
                    </div>
                </div>
            `;
        }
    } else if (result.events && typeof result.events === 'object' && !Array.isArray(result.events)) {
        // If events is a single object, try to extract action and jersey
        const action = result.events.action || result.events.event_type || '';
        const jerseyNumber = result.events.jersey_number;
        
        if (action) {
            const actionText = jerseyNumber ? `${action} #${jerseyNumber}` : action;
            eventsHtml = `
                <div class="result-events">
                    <div class="result-events-title">Action</div>
                    <div class="result-events-list">
                        <span class="result-event-tag">${escapeHtml(actionText)}</span>
                    </div>
                </div>
            `;
        }
    }
    
    // Get video URL from backend (already downloaded and served locally)
    const videoUrl = result.url ? (result.url.startsWith('http') ? result.url : `${API_BASE_URL}${result.url}`) : '';
    
    card.innerHTML = `
        <video class="result-video" preload="metadata" muted controls>
            <source src="${videoUrl}" type="video/mp4">
        </video>
        <div class="result-info">
            <div class="result-meta">
                <span class="result-score">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                    </svg>
                    ${(result.score || 0).toFixed(3)}
                </span>
                <span class="result-time">${timeRange}</span>
            </div>
            ${eventsHtml}
            <div class="result-video-id">Video ID: ${result.video_id || 'N/A'}</div>
        </div>
    `;
    
    // Add click handler to play video in the card itself
    // Note: We don't want to play on card click since video has controls
    // Users can click the video controls directly
    // But we'll still highlight the card
    card.addEventListener('click', (e) => {
        // Don't interfere if clicking on video controls
        if (e.target === videoElement || videoElement.contains(e.target)) {
            return;
        }
        
        // Highlight the selected card
        document.querySelectorAll('.result-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
    });
    
    // Get the video element
    const videoElement = card.querySelector('.result-video');
    
    // Handle video loading errors
    videoElement.addEventListener('error', (e) => {
        console.error('Video load error:', e);
        console.error('Video URL:', videoUrl);
        // Show error message or placeholder
        videoElement.style.background = '#f0f0f0';
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = 'display: flex; align-items: center; justify-content: center; height: 100%; color: #999;';
        errorDiv.textContent = 'Video unavailable';
        videoElement.parentNode.replaceChild(errorDiv, videoElement);
    });
    
    // Log when video metadata is loaded
    videoElement.addEventListener('loadedmetadata', () => {
        console.log('Video metadata loaded for URL:', videoUrl);
    });
    
    // Optional: Preview on hover (can be removed if not needed)
    let hoverTimeout;
    card.addEventListener('mouseenter', () => {
        // Small delay before preview to avoid accidental plays
        hoverTimeout = setTimeout(() => {
            if (videoElement.paused) {
                videoElement.play().catch(e => {
                    console.log('Preview play failed:', e);
                });
            }
        }, 300);
    });
    card.addEventListener('mouseleave', () => {
        clearTimeout(hoverTimeout);
        if (!videoElement.controls || document.activeElement !== videoElement) {
            videoElement.pause();
            videoElement.currentTime = 0;
        }
    });
    
    return card;
}

// Format time in seconds to MM:SS format
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Hide results container
function hideResults() {
    resultsContainer.style.display = 'none';
    resultsGrid.innerHTML = '';
    
    // Remove has-results class
    document.querySelector('.page-wrapper').classList.remove('has-results');
    
    // Remove scrolled class from search header
    const searchHeader = document.querySelector('.search-header');
    searchHeader.classList.remove('scrolled');
    
    // Restore background video
    backgroundVideo.style.display = 'block';
    document.body.style.background = '';
    
    // Scroll back to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

