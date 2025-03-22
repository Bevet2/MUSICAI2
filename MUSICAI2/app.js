// MUSICAI2 Frontend Logic

// API endpoints
const API = {
    SEARCH: '/api/search',
    GENRES: '/api/genres',
    REMIX: '/api/remix',
    CREATE: '/api/create',
    VOICE_STYLES: '/api/voice-styles'
};

// State management
let state = {
    mode: 'remix',
    selectedSongs: [],
    genres: [],
    voiceStyles: [],
    processing: false
};

// DOM Elements
const elements = {
    remixMode: document.getElementById('remixMode'),
    createMode: document.getElementById('createMode'),
    searchInput: document.getElementById('searchInput'),
    searchBtn: document.getElementById('searchBtn'),
    searchResults: document.getElementById('searchResults'),
    remixSection: document.getElementById('remixSection'),
    createSection: document.getElementById('createSection'),
    genreSelect: document.getElementById('genreSelect'),
    voiceSelect: document.getElementById('voiceSelect'),
    lyricsInput: document.getElementById('lyricsInput'),
    remixBtn: document.getElementById('remixBtn'),
    createBtn: document.getElementById('createBtn'),
    addSongBtn: document.getElementById('addSongBtn'),
    processingSection: document.getElementById('processingSection'),
    resultSection: document.getElementById('resultSection'),
    audioPlayer: document.getElementById('audioPlayer'),
    downloadBtn: document.getElementById('downloadBtn')
};

// Initialize application
async function init() {
    // Load available genres
    const genresResponse = await fetch(API.GENRES);
    const genresData = await genresResponse.json();
    state.genres = genresData.genres;
    
    // Load voice styles
    const stylesResponse = await fetch(API.VOICE_STYLES);
    const stylesData = await stylesResponse.json();
    state.voiceStyles = stylesData.styles;
    
    // Populate selects
    populateGenreSelect();
    populateVoiceSelect();
    
    // Setup event listeners
    setupEventListeners();
}

// Event listeners setup
function setupEventListeners() {
    // Mode switching
    elements.remixMode.addEventListener('click', () => switchMode('remix'));
    elements.createMode.addEventListener('click', () => switchMode('create'));
    
    // Search
    elements.searchBtn.addEventListener('click', handleSearch);
    elements.searchInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') handleSearch();
    });
    
    // Action buttons
    elements.remixBtn.addEventListener('click', handleRemix);
    elements.createBtn.addEventListener('click', handleCreate);
    elements.addSongBtn.addEventListener('click', () => {
        if (state.selectedSongs.length < 3) {
            handleSearch();
        }
    });
    
    // Download
    elements.downloadBtn.addEventListener('click', handleDownload);
}

// Mode switching
function switchMode(mode) {
    state.mode = mode;
    
    // Update UI
    elements.remixMode.classList.toggle('active', mode === 'remix');
    elements.createMode.classList.toggle('active', mode === 'create');
    elements.remixSection.classList.toggle('hidden', mode !== 'remix');
    elements.createSection.classList.toggle('hidden', mode !== 'create');
    
    // Clear selections
    state.selectedSongs = [];
    updateSelectedSongs();
}

// Populate select elements
function populateGenreSelect() {
    elements.genreSelect.innerHTML = state.genres
        .map(genre => `<option value="${genre}">${genre}</option>`)
        .join('');
}

function populateVoiceSelect() {
    elements.voiceSelect.innerHTML = state.voiceStyles
        .map(style => `<option value="${style}">${style}</option>`)
        .join('');
}

// Search handling
async function handleSearch() {
    const query = elements.searchInput.value.trim();
    if (!query) return;
    
    try {
        const response = await fetch(API.SEARCH, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, limit: 10 })
        });
        
        const data = await response.json();
        displaySearchResults(data.results);
        
    } catch (error) {
        console.error('Search error:', error);
        alert('Error searching for songs');
    }
}

// Display search results
function displaySearchResults(results) {
    elements.searchResults.innerHTML = results
        .map(result => `
            <div class="result-card" onclick="selectSong(${JSON.stringify(result).replace(/"/g, '&quot;')})">
                <img src="${result.thumbnail}" alt="${result.title}">
                <div class="info">
                    <h3>${result.title}</h3>
                    <p>${formatDuration(result.duration)}</p>
                </div>
            </div>
        `)
        .join('');
}

// Song selection
function selectSong(song) {
    if (state.mode === 'remix') {
        state.selectedSongs = [song];
    } else {
        if (state.selectedSongs.length < 3) {
            state.selectedSongs.push(song);
        }
    }
    
    updateSelectedSongs();
    elements.searchResults.innerHTML = '';
    elements.searchInput.value = '';
}

// Update selected songs display
function updateSelectedSongs() {
    const container = state.mode === 'remix' 
        ? elements.remixSection.querySelector('.selected-song')
        : elements.createSection.querySelector('.selected-songs');
        
    container.innerHTML = state.selectedSongs
        .map((song, index) => `
            <div class="song-item">
                <img src="${song.thumbnail}" alt="${song.title}">
                <div class="info">
                    <h3>${song.title}</h3>
                    <p>${formatDuration(song.duration)}</p>
                </div>
                <button class="remove-btn" onclick="removeSong(${index})">×</button>
            </div>
        `)
        .join('');
        
    // Update button states
    elements.remixBtn.disabled = state.selectedSongs.length !== 1;
    elements.createBtn.disabled = state.selectedSongs.length === 0;
    elements.addSongBtn.style.display = 
        state.selectedSongs.length >= 3 ? 'none' : 'block';
}

// Remove song from selection
function removeSong(index) {
    state.selectedSongs.splice(index, 1);
    updateSelectedSongs();
}

// Handle remix
async function handleRemix() {
    if (state.selectedSongs.length !== 1) return;
    
    const song = state.selectedSongs[0];
    const genre = elements.genreSelect.value;
    
    showProcessing(true);
    
    try {
        const response = await fetch(API.REMIX, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                youtube_url: song.url,
                target_genre: genre
            })
        });
        
        const data = await response.json();
        if (data.success) {
            showResult(data.output_path);
        } else {
            throw new Error(data.error);
        }
        
    } catch (error) {
        console.error('Remix error:', error);
        alert('Error processing song');
    } finally {
        showProcessing(false);
    }
}

// Handle create
async function handleCreate() {
    if (state.selectedSongs.length === 0) return;
    
    const lyrics = elements.lyricsInput.value.trim();
    const voiceStyle = elements.voiceSelect.value;
    
    showProcessing(true);
    
    try {
        const response = await fetch(API.CREATE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                youtube_urls: state.selectedSongs.map(song => song.url),
                lyrics,
                voice_style: voiceStyle
            })
        });
        
        const data = await response.json();
        if (data.success) {
            showResult(data.output_path);
        } else {
            throw new Error(data.error);
        }
        
    } catch (error) {
        console.error('Create error:', error);
        alert('Error creating song');
    } finally {
        showProcessing(false);
    }
}

// Handle download
function handleDownload() {
    const audioSrc = elements.audioPlayer.src;
    if (audioSrc) {
        const link = document.createElement('a');
        link.href = audioSrc;
        link.download = 'musicai2_output.mp3';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

// UI helpers
function showProcessing(show) {
    state.processing = show;
    elements.processingSection.classList.toggle('hidden', !show);
    elements.resultSection.classList.toggle('hidden', true);
}

function showResult(audioPath) {
    elements.processingSection.classList.toggle('hidden', true);
    elements.resultSection.classList.toggle('hidden', false);
    elements.audioPlayer.src = audioPath;
    elements.audioPlayer.load();
}

function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Initialize app
init();
