// Global state
var config = null;
var currentModelType = 'z_image';
var serverOnline = false;
var modelLoaded = false;
var modelFiles = {
diffusion: [],
vae: [],
llm: [],
clip: [],
t5: []
};

// DOM Elements cache
var elements = {};

// Initialize DOM element references
function initializeDOMElements() {
console.log('Initializing DOM elements...');

elements = {
// Status
serverStatus: document.getElementById('serverStatus'),
serverStatusText: document.getElementById('serverStatusText'),
currentModelText: document.getElementById('currentModelText'),
currentModelTypeText: document.getElementById('currentModelTypeText'),

// Model selection
modelType: document.getElementById('modelType'),
modelTypeInfo: document.getElementById('modelTypeInfo'),
modelConfigSection: document.getElementById('modelConfigSection'),

// Buttons
loadModelBtn: document.getElementById('loadModelBtn'),
unloadModelBtn: document.getElementById('unloadModelBtn'),
generateBtn: document.getElementById('generateBtn'),
videoGenerateBtn: document.getElementById('videoGenerateBtn'),
refreshGalleryBtn: document.getElementById('refreshGalleryBtn'),
refreshLogBtn: document.getElementById('refreshLogBtn'),
clearInitImageBtn: document.getElementById('clearInitImageBtn'),

// Generation settings
width: document.getElementById('width'),
height: document.getElementById('height'),
steps: document.getElementById('steps'),
stepsValue: document.getElementById('stepsValue'),
cfgScale: document.getElementById('cfgScale'),
cfgScaleValue: document.getElementById('cfgScaleValue'),
cfgGroup: document.getElementById('cfgGroup'),
guidance: document.getElementById('guidance'),
guidanceValue: document.getElementById('guidanceValue'),
guidanceGroup: document.getElementById('guidanceGroup'),
seed: document.getElementById('seed'),
sampler: document.getElementById('sampler'),
scheduler: document.getElementById('scheduler'),

// Advanced options
vaeTiling: document.getElementById('vaeTiling'),
offloadCpu: document.getElementById('offloadCpu'),
diffusionFa: document.getElementById('diffusionFa'),
flowShift: document.getElementById('flowShift'),

// Video options
videoOptionsSection: document.getElementById('videoOptionsSection'),
videoFrames: document.getElementById('videoFrames'),
fps: document.getElementById('fps'),
moeBoundary: document.getElementById('moeBoundary'),
moeBoundaryValue: document.getElementById('moeBoundaryValue'),

// Prompts
prompt: document.getElementById('prompt'),
negativePrompt: document.getElementById('negativePrompt'),
negativePromptGroup: document.getElementById('negativePromptGroup'),
videoPrompt: document.getElementById('videoPrompt'),
videoNegativePrompt: document.getElementById('videoNegativePrompt'),
videoNegativePromptGroup: document.getElementById('videoNegativePromptGroup'),

// File inputs
videoInitImage: document.getElementById('videoInitImage'),

// Output containers
outputContainer: document.getElementById('outputContainer'),
videoOutputContainer: document.getElementById('videoOutputContainer'),
loadingIndicator: document.getElementById('loadingIndicator'),
videoLoadingIndicator: document.getElementById('videoLoadingIndicator'),

// Other
messageArea: document.getElementById('messageArea'),
galleryGrid: document.getElementById('galleryGrid'),
serverLog: document.getElementById('serverLog'),
imageModal: document.getElementById('imageModal'),
modalImage: document.getElementById('modalImage'),
modalClose: document.getElementById('modalClose'),

// Settings
threads: document.getElementById('threads'),
serverPort: document.getElementById('serverPort'),
loraDir: document.getElementById('loraDir'),
embdDir: document.getElementById('embdDir')
};

console.log('DOM elements initialized');
}

// Fetch config and initialize app
async function initializeApp() {
console.log('Starting app initialization...');
initializeDOMElements();

try {
console.log('Fetching config...');
var response = await fetch('/api/config');
if (!response.ok) {
throw new Error('Failed to fetch config: ' + response.status);
}
config = await response.json();
console.log('Config loaded:', config);

populateModelTypes();
populateSamplers();
populateSchedulers();
applyDefaults();
setupEventListeners();
setupTabListeners();
loadModelList();
checkServerStatus();

// Poll server status every 5 seconds
setInterval(checkServerStatus, 5000);

console.log('App initialization complete');
} catch (error) {
console.error('Error initializing app:', error);
showMessage('Failed to load configuration: ' + error.message, 'error');
}
}

// Setup tab switching
function setupTabListeners() {
console.log('Setting up tab listeners...');

var tabBtns = document.querySelectorAll('.tab-btn');
console.log('Found ' + tabBtns.length + ' tab buttons');

tabBtns.forEach(function(btn) {
btn.addEventListener('click', function(e) {
e.preventDefault();
console.log('Tab clicked:', btn.dataset.tab);

// Remove active from all buttons
tabBtns.forEach(function(b) {
b.classList.remove('active');
});

// Remove active from all tab content
var tabContents = document.querySelectorAll('.tab-content');
tabContents.forEach(function(c) {
c.classList.remove('active');
});

// Add active to clicked button
btn.classList.add('active');

// Add active to corresponding tab content
var tabId = btn.dataset.tab + '-tab';
var tabContent = document.getElementById(tabId);
if (tabContent) {
tabContent.classList.add('active');
console.log('Activated tab:', tabId);
} else {
console.error('Tab content not found:', tabId);
}

// Load gallery if gallery tab selected
if (btn.dataset.tab === 'gallery') {
loadGallery();
}
});
});
}

// Populate model type dropdown
function populateModelTypes() {
console.log('Populating model types...');

if (!elements.modelType) {
console.error('modelType element not found');
return;
}

elements.modelType.innerHTML = '';

var modelTypes = Object.keys(config).filter(function(key) {
return key !== 'all';
});

console.log('Model types:', modelTypes);

modelTypes.forEach(function(modelType) {
var option = document.createElement('option');
option.value = modelType;
option.textContent = config[modelType].name || modelType;
elements.modelType.appendChild(option);
});

currentModelType = elements.modelType.value;
console.log('Current model type:', currentModelType);
}

// Populate samplers dropdown
function populateSamplers() {
if (!elements.sampler) return;

elements.sampler.innerHTML = '';

var samplers = config.all && config.all.samplers ? config.all.samplers : [];
samplers.forEach(function(sampler) {
var option = document.createElement('option');
option.value = sampler;
option.textContent = sampler;
elements.sampler.appendChild(option);
});
}

// Populate schedulers dropdown
function populateSchedulers() {
if (!elements.scheduler) return;

elements.scheduler.innerHTML = '';

var schedulers = config.all && config.all.schedulers ? config.all.schedulers : [];
schedulers.forEach(function(scheduler) {
var option = document.createElement('option');
option.value = scheduler;
option.textContent = scheduler;
elements.scheduler.appendChild(option);
});
}

// Apply defaults based on selected model type
function applyDefaults() {
console.log('Applying defaults for:', currentModelType);

var modelConfig = config[currentModelType];
var allDefaults = config.all && config.all.defaults ? config.all.defaults : {};

// Merge all defaults with model-specific defaults
var defaults = Object.assign({}, allDefaults);
if (modelConfig && modelConfig.defaults) {
defaults = Object.assign({}, defaults, modelConfig.defaults);
}

var params = modelConfig && modelConfig.params ? modelConfig.params : {};

console.log('Merged defaults:', defaults);
console.log('Params:', params);

// Update description
if (elements.modelTypeInfo) {
if (modelConfig && modelConfig.description) {
elements.modelTypeInfo.textContent = modelConfig.description;
} else {
elements.modelTypeInfo.textContent = '';
}
}

// Apply generation settings
if (elements.width && defaults.width !== undefined) {
elements.width.value = defaults.width;
}
if (elements.height && defaults.height !== undefined) {
elements.height.value = defaults.height;
}

if (elements.steps && defaults.steps !== undefined) {
elements.steps.value = defaults.steps;
if (elements.stepsValue) {
elements.stepsValue.textContent = defaults.steps;
}
}

if (elements.cfgScale && defaults.CFG_scale !== undefined) {
elements.cfgScale.value = defaults.CFG_scale;
if (elements.cfgScaleValue) {
elements.cfgScaleValue.textContent = defaults.CFG_scale;
}
}

if (elements.guidance && defaults.guidance !== undefined) {
elements.guidance.value = defaults.guidance;
if (elements.guidanceValue) {
elements.guidanceValue.textContent = defaults.guidance;
}
}

if (elements.seed && defaults.seed !== undefined) {
elements.seed.value = defaults.seed;
}

if (elements.sampler && defaults.sampler !== undefined) {
elements.sampler.value = defaults.sampler;
}

if (elements.scheduler && defaults.scheduler !== undefined) {
elements.scheduler.value = defaults.scheduler;
}

// Advanced options
if (elements.vaeTiling && defaults.VAE_tiling !== undefined) {
elements.vaeTiling.checked = defaults.VAE_tiling;
}
if (elements.offloadCpu && defaults['Offload-to-CPU'] !== undefined) {
elements.offloadCpu.checked = defaults['Offload-to-CPU'];
}
if (elements.diffusionFa && defaults.FA !== undefined) {
elements.diffusionFa.checked = defaults.FA;
}
if (elements.flowShift) {
if (defaults.flow_shift !== undefined && defaults.flow_shift !== 'auto') {
elements.flowShift.value = defaults.flow_shift;
} else {
elements.flowShift.value = '';
}
}

// Video options
if (elements.videoFrames && defaults.videoFrames !== undefined) {
elements.videoFrames.value = defaults.videoFrames;
}
if (elements.fps && defaults.fps !== undefined) {
elements.fps.value = defaults.fps;
}
if (elements.moeBoundary && defaults.moe_boundary !== undefined) {
elements.moeBoundary.value = defaults.moe_boundary;
if (elements.moeBoundaryValue) {
elements.moeBoundaryValue.textContent = defaults.moe_boundary;
}
}

// Update visibility based on params
updateVisibility(params, defaults);

// Update model selection UI
updateModelSelectionUI();
}

// Update visibility of elements
function updateVisibility(params, defaults) {
// CFG visibility
if (elements.cfgGroup) {
if (params.enable_CFG === false) {
elements.cfgGroup.classList.add('hidden');
} else {
elements.cfgGroup.classList.remove('hidden');
}
}

// Negative prompt visibility
var hideNegPrompt = params.enable_negative_prompt === false;

if (elements.negativePromptGroup) {
if (hideNegPrompt) {
elements.negativePromptGroup.classList.add('hidden');
} else {
elements.negativePromptGroup.classList.remove('hidden');
}
}
if (elements.videoNegativePromptGroup) {
if (hideNegPrompt) {
elements.videoNegativePromptGroup.classList.add('hidden');
} else {
elements.videoNegativePromptGroup.classList.remove('hidden');
}
}

// Guidance visibility
if (elements.guidanceGroup) {
if (defaults.showGuidance === true) {
elements.guidanceGroup.classList.remove('hidden');
} else {
elements.guidanceGroup.classList.add('hidden');
}
}

// Video options visibility (only for wan)
if (elements.videoOptionsSection) {
if (currentModelType === 'wan') {
elements.videoOptionsSection.classList.remove('hidden');
} else {
elements.videoOptionsSection.classList.add('hidden');
}
}
}

// Update model selection UI based on model type
function updateModelSelectionUI() {
console.log('Updating model selection UI...');

if (!elements.modelConfigSection) {
console.error('modelConfigSection not found');
return;
}

var modelConfig = config[currentModelType];
var models = modelConfig && modelConfig.models ? modelConfig.models : {};

console.log('Models for', currentModelType, ':', models);

var html = '';

// Diffusion model (always shown if exists)
if (models.diffusion !== undefined) {
html += createModelSelectHTML('diffusionModel', 'Diffusion Model', models.diffusion, true);
}

// LLM model (z_image and wan)
if (models.llm !== undefined) {
html += createModelSelectHTML('llmModel', 'LLM Model', models.llm, true);
}

// CLIP-L model
if (models.clip_l !== undefined) {
html += createModelSelectHTML('clipLModel', 'CLIP-L Model', models.clip_l, false);
}

// CLIP-G model (SD3 only)
if (models.clip_g !== undefined) {
html += createModelSelectHTML('clipGModel', 'CLIP-G Model', models.clip_g, false);
}

// T5-XXL model
if (models.t5xxl !== undefined) {
html += createModelSelectHTML('t5xxlModel', 'T5-XXL Model', models.t5xxl, false);
}

// VAE model
var vaeModel = models.vae || models.vae_model;
if (vaeModel !== undefined) {
html += createModelSelectHTML('vaeModel', 'VAE Model', vaeModel, false);
}

elements.modelConfigSection.innerHTML = html;

// Now populate the dropdowns with actual files
populateModelDropdowns();
}

// Create HTML for model select dropdown
function createModelSelectHTML(id, label, defaultValue, required) {
var requiredClass = required ? ' required' : '';
var requiredStar = required ? '<span style="color: #ff6b6b;"> *</span>' : '';
var defaultText = required ? 'Select a model...' : 'None (use embedded)';

return '<div class="form-group' + requiredClass + '">' +
'<label for="' + id + '">' + label + requiredStar + '<span class="model-count" id="' + id + 'Count"></span></label>' +
'<select id="' + id + '">' +
'<option value="">' + defaultText + '</option>' +
'</select>' +
'<div class="info-text">Default: ' + (defaultValue || 'None') + '</div>' +
'</div>';
}

// Load available models from server
async function loadModelList() {
console.log('Loading model list from server...');

try {
var response = await fetch('/list_models');
var data = await response.json();

console.log('Model list received:', data);

modelFiles.diffusion = data.diffusion || [];
modelFiles.vae = data.vae || [];
modelFiles.llm = data.llm || [];
modelFiles.clip = data.clip || [];
modelFiles.t5 = data.t5 || [];

populateModelDropdowns();
} catch (error) {
console.error('Failed to load model list:', error);
}
}

// Populate model dropdowns with available files
function populateModelDropdowns() {
console.log('Populating model dropdowns...');

var modelConfig = config[currentModelType];
var defaultModels = modelConfig && modelConfig.models ? modelConfig.models : {};

populateModelSelect('diffusionModel', modelFiles.diffusion, defaultModels.diffusion);
populateModelSelect('llmModel', modelFiles.llm, defaultModels.llm);
populateModelSelect('clipLModel', modelFiles.clip, defaultModels.clip_l);
populateModelSelect('clipGModel', modelFiles.clip, defaultModels.clip_g);
populateModelSelect('t5xxlModel', modelFiles.t5, defaultModels.t5xxl);
populateModelSelect('vaeModel', modelFiles.vae, defaultModels.vae || defaultModels.vae_model);
}

// Populate a single model select dropdown
function populateModelSelect(selectId, models, defaultModel) {
var selectEl = document.getElementById(selectId);
var countEl = document.getElementById(selectId + 'Count');

if (!selectEl) {
return;
}

// Save first option
var firstOptionText = selectEl.options.length > 0 ? selectEl.options[0].textContent : 'Select...';

selectEl.innerHTML = '';

// Add default empty option
var defaultOption = document.createElement('option');
defaultOption.value = '';
defaultOption.textContent = firstOptionText;
selectEl.appendChild(defaultOption);

// Add model options
models.forEach(function(model) {
var option = document.createElement('option');
option.value = model.path;
option.textContent = model.name;
if (model.size_human) {
option.textContent += ' (' + model.size_human + ')';
}
selectEl.appendChild(option);
});

// Try to select default model
if (defaultModel) {
for (var i = 0; i < selectEl.options.length; i++) {
var optionValue = selectEl.options[i].value;
if (optionValue && optionValue.indexOf(defaultModel) !== -1) {
selectEl.selectedIndex = i;
break;
}
}
}

// Update count
if (countEl) {
countEl.textContent = '(' + models.length + ')';
}
}

// Setup event listeners
function setupEventListeners() {
console.log('Setting up event listeners...');

// Model type change
if (elements.modelType) {
elements.modelType.addEventListener('change', function() {
currentModelType = this.value;
console.log('Model type changed to:', currentModelType);
applyDefaults();
});
}

// Slider updates
if (elements.steps) {
elements.steps.addEventListener('input', function() {
if (elements.stepsValue) {
elements.stepsValue.textContent = this.value;
}
});
}

if (elements.cfgScale) {
elements.cfgScale.addEventListener('input', function() {
if (elements.cfgScaleValue) {
elements.cfgScaleValue.textContent = this.value;
}
});
}

if (elements.guidance) {
elements.guidance.addEventListener('input', function() {
if (elements.guidanceValue) {
elements.guidanceValue.textContent = this.value;
}
});
}

if (elements.moeBoundary) {
elements.moeBoundary.addEventListener('input', function() {
if (elements.moeBoundaryValue) {
elements.moeBoundaryValue.textContent = this.value;
}
});
}

if (elements.clearInitImageBtn) {
    elements.clearInitImageBtn.addEventListener('click', clearInitImage);
}

// Show/hide strength slider when init image is selected
var txt2imgInitImage = document.getElementById('txt2imgInitImage');
var txt2imgStrengthGroup = document.getElementById('txt2imgStrengthGroup');
var txt2imgStrength = document.getElementById('txt2imgStrength');
var txt2imgStrengthValue = document.getElementById('txt2imgStrengthValue');

if (txt2imgInitImage && txt2imgStrengthGroup) {
txt2imgInitImage.addEventListener('change', function() {
if (this.files && this.files[0]) {
txt2imgStrengthGroup.style.display = 'block';
if (txt2imgStrengthValue && txt2imgStrength) {
txt2imgStrengthValue.textContent = txt2imgStrength.value;
}
} else {
txt2imgStrengthGroup.style.display = 'none';
}
});
}

if (txt2imgStrength && txt2imgStrengthValue) {
txt2imgStrength.addEventListener('input', function() {
txt2imgStrengthValue.textContent = this.value;
});
}

// Model loading
if (elements.loadModelBtn) {
elements.loadModelBtn.addEventListener('click', loadModel);
}

if (elements.unloadModelBtn) {
elements.unloadModelBtn.addEventListener('click', unloadModel);
}

// Generation
if (elements.generateBtn) {
elements.generateBtn.addEventListener('click', function() {
generateImage();
});
}

if (elements.videoGenerateBtn) {
elements.videoGenerateBtn.addEventListener('click', generateVideo);
}

// Gallery
if (elements.refreshGalleryBtn) {
elements.refreshGalleryBtn.addEventListener('click', loadGallery);
}

// Log
if (elements.refreshLogBtn) {
elements.refreshLogBtn.addEventListener('click', loadServerLog);
}

// Modal
if (elements.modalClose) {
elements.modalClose.addEventListener('click', function() {
if (elements.imageModal) {
elements.imageModal.classList.remove('active');

// Pause and reset video if it exists
var videoModal = document.getElementById('modalVideo');
if (videoModal) {
videoModal.pause();
videoModal.currentTime = 0;
}
}
});
}

if (elements.imageModal) {
elements.imageModal.addEventListener('click', function(e) {
if (e.target === elements.imageModal) {
elements.imageModal.classList.remove('active');

// Pause and reset video if it exists
var videoModal = document.getElementById('modalVideo');
if (videoModal) {
videoModal.pause();
videoModal.currentTime = 0;
}
}
});
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
if (e.key === 'Escape' && elements.imageModal) {
elements.imageModal.classList.remove('active');
}

if (e.ctrlKey && e.key === 'Enter') {
var activeTab = document.querySelector('.tab-content.active');
if (activeTab) {
if (activeTab.id === 'txt2img-tab' && elements.generateBtn) {
elements.generateBtn.click();
} else if (activeTab.id === 'video-tab' && elements.videoGenerateBtn) {
elements.videoGenerateBtn.click();
}
}
}
});

console.log('Event listeners set up complete');
}

async function checkServerStatus() {
    try {
        var response = await fetch('/server_status');
        var data = await response.json();

        serverOnline = data.server_responsive;
        modelLoaded = data.process_running && data.server_responsive;

        // ALWAYS UPDATE MODEL INDICATORS
        if (elements.currentModelText) {
            elements.currentModelText.textContent = data.current_model || 'None';
        }
        if (elements.currentModelTypeText) {
            elements.currentModelTypeText.textContent = data.model_type || '-';
        }

        // YOUR PERFECT STATE MACHINE:
        const serverModeActive = data.process_running || data.server_responsive;
        
        // 1. FREEZE GUI ELEMENTS when process_running (loading OR loaded)
        document.querySelectorAll('#modelType, #diffusionModel, #vaeModel, #llmModel, #clipLModel, #clipGModel, #t5xxlModel, #seed, #sampler, #scheduler, #threads, #vaeTiling, #offloadCpu, #diffusionFa, #flowShift').forEach(el => {
            el.disabled = serverModeActive;
            el.style.opacity = serverModeActive ? '0.5' : '1';
        });

        // 2. SERVER STATUS LOGIC
        if (elements.serverStatus && elements.serverStatusText) {
            if (data.process_running && data.server_responsive) {
                // BOTH true → "Online"
                elements.serverStatus.classList.add('active');
                elements.serverStatusText.textContent = 'Online';
            } else if (!data.process_running) {
                // process_running = false → "Offline" 
                elements.serverStatus.classList.remove('active');
                elements.serverStatusText.textContent = 'Offline';
            }
            // process_running=true but !server_responsive → "Starting..." (don't touch)
        }

    } catch (error) {
        if (elements.serverStatus) {
            elements.serverStatus.classList.remove('active');
        }
        if (elements.serverStatusText) {
            elements.serverStatusText.textContent = 'Error';
        }
        serverOnline = false;
        modelLoaded = false;
    }
}


// Build model args for loading
function buildModelArgs() {
var args = {
model_type: currentModelType,
threads: elements.threads ? parseInt(elements.threads.value) : 4,
port: elements.serverPort ? parseInt(elements.serverPort.value) : 8080
};

var diffusionModel = document.getElementById('diffusionModel');
var llmModel = document.getElementById('llmModel');
var clipLModel = document.getElementById('clipLModel');
var clipGModel = document.getElementById('clipGModel');
var t5xxlModel = document.getElementById('t5xxlModel');
var vaeModel = document.getElementById('vaeModel');

if (diffusionModel && diffusionModel.value) args.diffusion_model = diffusionModel.value;
if (llmModel && llmModel.value) args.llm = llmModel.value;
if (clipLModel && clipLModel.value) args.clip_l = clipLModel.value;
if (clipGModel && clipGModel.value) args.clip_g = clipGModel.value;
if (t5xxlModel && t5xxlModel.value) args.t5xxl = t5xxlModel.value;
if (vaeModel && vaeModel.value) args.vae = vaeModel.value;

// Advanced options
if (elements.vaeTiling && elements.vaeTiling.checked) args.vae_tiling = true;
if (elements.offloadCpu && elements.offloadCpu.checked) args.offload_to_cpu = true;
if (elements.diffusionFa && elements.diffusionFa.checked) args.diffusion_fa = true;
if (elements.flowShift && elements.flowShift.value) args.flow_shift = parseFloat(elements.flowShift.value);
if (elements.loraDir && elements.loraDir.value) args.lora_model_dir = elements.loraDir.value;
if (elements.embdDir && elements.embdDir.value) args.embd_dir = elements.embdDir.value;

// Video options
if (currentModelType === 'wan' && elements.moeBoundary) {
args.moe_boundary = parseFloat(elements.moeBoundary.value);
}

return args;
}

// Load model
async function loadModel() {
var args = buildModelArgs();

console.log('Loading model with args:', args);

if (!args.diffusion_model) {
showMessage('Please select a diffusion model', 'error');
return;
}

if ((currentModelType === 'z_image' || currentModelType === 'wan') && !args.llm) {
showMessage('Please select an LLM model (required for ' + currentModelType + ')', 'error');
return;
}

if (elements.loadModelBtn) {
elements.loadModelBtn.disabled = true;
elements.loadModelBtn.textContent = 'Loading...';
}

try {
var response = await fetch('/load_model', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify(args)
});

var data = await response.json();

if (response.ok) {
showMessage('Model loaded successfully', 'success');
checkServerStatus();
} else {
showMessage(data.error || 'Failed to load model', 'error');
}
} catch (error) {
showMessage('Failed to load model: ' + error.message, 'error');
} finally {
if (elements.loadModelBtn) {
elements.loadModelBtn.disabled = false;
elements.loadModelBtn.textContent = 'Load Model';
}
}
}

// Unload model
async function unloadModel() {
try {
var response = await fetch('/unload_model', { method: 'POST' });
var data = await response.json();

if (response.ok) {
showMessage('Model unloaded', 'success');
checkServerStatus();
} else {
showMessage(data.error || 'Failed to unload model', 'error');
}
} catch (error) {
showMessage('Failed to unload model: ' + error.message, 'error');
}
}

// Build generation parameters
function buildGenerationParams() {
var params = {
prompt: elements.prompt ? elements.prompt.value : '',
negative_prompt: elements.negativePrompt ? elements.negativePrompt.value : '',
width: elements.width ? parseInt(elements.width.value) : 512,
height: elements.height ? parseInt(elements.height.value) : 512,
steps: elements.steps ? parseInt(elements.steps.value) : 20,
cfg_scale: elements.cfgScale ? parseFloat(elements.cfgScale.value) : 7.0,
seed: elements.seed ? parseInt(elements.seed.value) : -1,
sampler: elements.sampler ? elements.sampler.value : 'euler',
scheduler: elements.scheduler ? elements.scheduler.value : 'normal'
};

var modelConfig = config[currentModelType];
if (modelConfig && modelConfig.defaults && modelConfig.defaults.showGuidance && elements.guidance) {
params.guidance = parseFloat(elements.guidance.value);
}

return params;
}


// Generate image - FIXED VERSION
async function generateImage() {
    var prompt = elements.prompt ? elements.prompt.value.trim() : '';

    if (!prompt) {
        showMessage('Please enter a prompt', 'error');
        return;
    }

    var txt2imgInitImage = document.getElementById('txt2imgInitImage');
    var hasInitImage = txt2imgInitImage && txt2imgInitImage.files && txt2imgInitImage.files[0];

    if (elements.generateBtn) elements.generateBtn.disabled = true;
    if (elements.loadingIndicator) elements.loadingIndicator.classList.add('active');
    if (elements.outputContainer) elements.outputContainer.innerHTML = '';

    // *** Set server status to "Busy" with orange dot ***
    console.log('=== GENERATION STARTING ===');
    setServerBusyStatus(true);
    console.log('isGenerating is now:', isGenerating);

    var params = buildGenerationParams();

    console.log('Generating image with params:', params);

    try {
        // If there's an init image, upload it first
        if (hasInitImage) {
            var formData = new FormData();
            formData.append('image', txt2imgInitImage.files[0]);

            var uploadResponse = await fetch('/upload_init_image', {
                method: 'POST',
                body: formData
            });

            var uploadData = await uploadResponse.json();

            if (uploadResponse.ok && uploadData.path) {
                params.init_img = uploadData.path;

                // Add strength
                var txt2imgStrength = document.getElementById('txt2imgStrength');
                if (txt2imgStrength) {
                    params.strength = parseFloat(txt2imgStrength.value);
                }
            } else {
                throw new Error(uploadData.error || 'Failed to upload init image');
            }
        }

        var endpoint = '/generate';

        if (!serverOnline || !modelLoaded) {
            params.use_cli = true;
            params.model_args = buildModelArgs();
        }

        console.log('Using endpoint:', endpoint);

        // *** FIX 1: Increase timeout significantly for long-running operations ***
        var controller = new AbortController();
        var timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout

        var response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        // *** FIX 2: Better error handling - check if response is valid JSON ***
        var data;
        try {
            data = await response.json();
            console.log('Parsed response data:', data);
        } catch (jsonError) {
            console.error('JSON parse error:', jsonError);
            console.log('Response status:', response.status);
            
            // Try to get response text for debugging
            var responseClone = response.clone();
            var responseText = await responseClone.text();
            console.log('Response text (first 500 chars):', responseText.substring(0, 500));
            
            // If generation might still be running, poll for results
            showMessage('Generation started but response was invalid. Checking gallery...', 'info');
            
            // Wait a bit then check gallery
            setTimeout(async () => {
                await loadGallery();
                var galleryResponse = await fetch('/list_outputs');
                var galleryData = await galleryResponse.json();
                
                if (galleryData.files && galleryData.files.length > 0) {
                    // Get most recent file
                    var latestFile = galleryData.files[0];
                    
                    var img = document.createElement('img');
                    img.src = latestFile.url;
                    img.addEventListener('click', function() {
                        openModal(img.src);
                    });
                    if (elements.outputContainer) {
                        elements.outputContainer.innerHTML = '';
                        elements.outputContainer.appendChild(img);
                    }
                    showMessage('Image generated successfully (recovered from gallery)', 'success');
                } else {
                    if (elements.outputContainer) {
                        elements.outputContainer.innerHTML = '<div class="placeholder"><div class="placeholder-icon">[X]</div><p>No image found</p></div>';
                    }
                }
            }, 2000);
            
            return; // Exit early
        }

        // *** FIX 3: Handle successful generation ***
        if (response.ok && (data.status === 'success' || data.image)) {
            console.log('Generation successful! Response data:', data);
            var img = document.createElement('img');
            if (data.image) {
                img.src = 'data:image/png;base64,' + data.image;
                console.log('Using base64 image from response');
            } else if (data.output) {
                img.src = '/output/' + data.output;
                console.log('Using output filename:', data.output);
            } else {
                console.error('No image data found in response:', data);
                throw new Error('No image data in response');
            }
            img.addEventListener('click', function() {
                openModal(img.src);
            });
            if (elements.outputContainer) {
                elements.outputContainer.innerHTML = '';
                elements.outputContainer.appendChild(img);
            }
            showMessage(data.message || 'Image generated successfully', 'success');
        } else {
            console.error('Generation failed. Response:', data);
            showMessage(data.message || data.error || 'Failed to generate image', 'error');
            if (elements.outputContainer) {
                elements.outputContainer.innerHTML = '<div class="placeholder"><div class="placeholder-icon">[X]</div><p>Generation failed</p></div>';
            }
        }
    } catch (error) {
        // *** FIX 4: Distinguish between timeout and actual errors ***
        if (error.name === 'AbortError') {
            showMessage('Generation timed out. Check gallery for results...', 'warning');
            
            // Poll gallery after timeout
            setTimeout(async () => {
                await loadGallery();
            }, 2000);
        } else {
            showMessage('Failed to generate image: ' + error.message, 'error');
            if (elements.outputContainer) {
                elements.outputContainer.innerHTML = '<div class="placeholder"><div class="placeholder-icon">[X]</div><p>Generation failed</p></div>';
            }
        }
    } finally {
        if (elements.generateBtn) elements.generateBtn.disabled = false;
        if (elements.loadingIndicator) elements.loadingIndicator.classList.remove('active');
        
        // *** Clear busy status ***
        console.log('=== GENERATION COMPLETE ===');
        console.log('isGenerating before clear:', isGenerating);
        setServerBusyStatus(false);
        console.log('isGenerating after clear:', isGenerating);
    }
}

// *** NEW: Set server busy status ***
function setServerBusyStatus(isBusy) {
    console.log('setServerBusyStatus called with isBusy =', isBusy); // *** Debug ***
    isGenerating = isBusy; // *** Set the global flag ***
    
    if (elements.serverStatus && elements.serverStatusText) {
        if (isBusy) {
            console.log('Setting status to Busy (orange)'); // *** Debug ***
            elements.serverStatus.classList.remove('active');
            elements.serverStatus.classList.add('busy');
            elements.serverStatusText.textContent = 'Busy';
        } else {
            console.log('Clearing busy status, calling checkServerStatus'); // *** Debug ***
            // Remove busy class and restore to online/offline based on actual status
            elements.serverStatus.classList.remove('busy');
            // Force an immediate status check to restore proper state
            checkServerStatus();
        }
    }
}

// *** FIX 5: Update showMessage to support 'warning' and 'info' types ***
function showMessage(message, type) {
    type = type || 'error';

    if (!elements.messageArea) {
        console.log('Message (' + type + '):', message);
        return;
    }

    var className;
    switch(type) {
        case 'error':
            className = 'error-message';
            break;
        case 'success':
            className = 'success-message';
            break;
        case 'warning':
            className = 'warning-message';
            break;
        case 'info':
            className = 'info-message';
            break;
        default:
            className = 'error-message';
    }
    
    elements.messageArea.innerHTML = '<div class="' + className + '">' + message + '</div>';

    setTimeout(function() {
        if (elements.messageArea) {
            elements.messageArea.innerHTML = '';
        }
    }, 5000);
}


// File to base64
function fileToBase64(file) {
return new Promise(function(resolve, reject) {
var reader = new FileReader();
reader.onload = function() {
resolve(reader.result);
};
reader.onerror = reject;
reader.readAsDataURL(file);
});
}

// Generate video
async function generateVideo() {
var prompt = elements.videoPrompt ? elements.videoPrompt.value.trim() : '';

if (!prompt) {
showMessage('Please enter a prompt', 'error');
return;
}

if (currentModelType !== 'wan') {
showMessage('Video generation requires Wan model type. Please select Wan and load a model.', 'error');
return;
}

if (elements.videoGenerateBtn) elements.videoGenerateBtn.disabled = true;
if (elements.videoLoadingIndicator) elements.videoLoadingIndicator.classList.add('active');
if (elements.videoOutputContainer) elements.videoOutputContainer.innerHTML = '';

var params = {
prompt: elements.videoPrompt.value,
negative_prompt: elements.videoNegativePrompt ? elements.videoNegativePrompt.value : '',
width: elements.width ? parseInt(elements.width.value) : 512,
height: elements.height ? parseInt(elements.height.value) : 512,
steps: elements.steps ? parseInt(elements.steps.value) : 20,
cfg_scale: elements.cfgScale ? parseFloat(elements.cfgScale.value) : 7.0,
seed: elements.seed ? parseInt(elements.seed.value) : -1,
sampler: elements.sampler ? elements.sampler.value : 'euler',
scheduler: elements.scheduler ? elements.scheduler.value : 'normal',
video_frames: elements.videoFrames ? parseInt(elements.videoFrames.value) : 16,
fps: elements.fps ? parseInt(elements.fps.value) : 24
};

try {
var fileInput = elements.videoInitImage;
if (fileInput && fileInput.files && fileInput.files[0]) {
params.init_image = await fileToBase64(fileInput.files[0]);
}

var endpoint = serverOnline && modelLoaded ? '/generate_video' : '/generate_video_cli';

if (!serverOnline || !modelLoaded) {
params.use_cli = true;
params.model_args = buildModelArgs();
}

var response = await fetch(endpoint, {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify(params)
});

var data = await response.json();

if (response.ok && data.video) {
var video = document.createElement('video');
video.src = 'data:video/mp4;base64,' + data.video;
video.controls = true;
video.autoplay = true;
video.loop = true;
video.style.maxWidth = '100%';
video.style.maxHeight = '600px';
if (elements.videoOutputContainer) {
elements.videoOutputContainer.innerHTML = '';
elements.videoOutputContainer.appendChild(video);
}
showMessage('Video generated successfully', 'success');
} else {
showMessage(data.error || 'Failed to generate video', 'error');
if (elements.videoOutputContainer) {
elements.videoOutputContainer.innerHTML = '<div class="placeholder"><div class="placeholder-icon">[X]</div><p>Generation failed</p></div>';
}
}
} catch (error) {
showMessage('Failed to generate video: ' + error.message, 'error');
if (elements.videoOutputContainer) {
elements.videoOutputContainer.innerHTML = '<div class="placeholder"><div class="placeholder-icon">[X]</div><p>Generation failed</p></div>';
}
} finally {
if (elements.videoGenerateBtn) elements.videoGenerateBtn.disabled = false;
if (elements.videoLoadingIndicator) elements.videoLoadingIndicator.classList.remove('active');
}
}

// Load gallery
async function loadGallery() {
console.log('Loading gallery...');

if (!elements.galleryGrid) {
console.error('Gallery grid not found');
return;
}

try {
var response = await fetch('/list_outputs');
var data = await response.json();

console.log('Gallery data:', data);

elements.galleryGrid.innerHTML = '';

if (data.files && data.files.length > 0) {
data.files.forEach(function(file) {
var div = document.createElement('div');
div.className = 'gallery-item';

if (file.type === 'video') {
// Create video element for videos
var video = document.createElement('video');
video.src = file.url;
video.style.width = '100%';
video.style.height = '100%';
video.style.objectFit = 'cover';
video.muted = true;
video.loop = true;

// Play preview on hover
div.addEventListener('mouseenter', function() {
video.play();
});
div.addEventListener('mouseleave', function() {
video.pause();
video.currentTime = 0;
});

// Click to open in modal
video.addEventListener('click', function(e) {
e.stopPropagation();
openVideoModal(file.url, file.name);
});

div.appendChild(video);

// Add video indicator badge
var badge = document.createElement('div');
badge.style.position = 'absolute';
badge.style.top = '5px';
badge.style.right = '5px';
badge.style.background = 'rgba(0, 0, 0, 0.7)';
badge.style.color = 'white';
badge.style.padding = '2px 6px';
badge.style.borderRadius = '4px';
badge.style.fontSize = '0.7rem';
badge.textContent = ' VIDEO';
div.style.position = 'relative';
div.appendChild(badge);
} else {
// Create image element for images
var img = document.createElement('img');
img.src = file.url;
img.alt = file.name;
img.addEventListener('click', function() {
openModal(file.url);
});
div.appendChild(img);
}

elements.galleryGrid.appendChild(div);
});
} else {
elements.galleryGrid.innerHTML = '<p>No images or videos generated yet</p>';
}
} catch (error) {
console.error('Failed to load gallery:', error);
elements.galleryGrid.innerHTML = '<p>Failed to load gallery</p>';
}
}

// Add new function for video modal
function openVideoModal(src, name) {
if (!elements.imageModal) return;

// Clear existing content
var modalContent = elements.imageModal;

// Hide the image if it exists
if (elements.modalImage) {
elements.modalImage.style.display = 'none';
}

// Create or get video element
var videoModal = document.getElementById('modalVideo');
if (!videoModal) {
videoModal = document.createElement('video');
videoModal.id = 'modalVideo';
videoModal.controls = true;
videoModal.autoplay = true;
videoModal.loop = true;
videoModal.style.maxWidth = '90%';
videoModal.style.maxHeight = '90%';
videoModal.style.objectFit = 'contain';
modalContent.appendChild(videoModal);
}

videoModal.style.display = 'block';
videoModal.src = src;
modalContent.classList.add('active');
}

// Load server log
async function loadServerLog() {
if (!elements.serverLog) return;

try {
var response = await fetch('/server_log?lines=100');
var data = await response.json();
elements.serverLog.textContent = data.log || 'No log data';
} catch (error) {
elements.serverLog.textContent = 'Failed to load log';
}
}

// Clear the init image input and reset strength slider
function clearInitImage() {
    const input = document.getElementById('txt2imgInitImage');
    const strengthGroup = document.getElementById('txt2imgStrengthGroup');

    if (input) {
        input.value = ''; // Clear file input
    }

    if (strengthGroup) {
        strengthGroup.style.display = 'none'; // Hide strength slider if visible
    }

    // Optional: Reset the strength slider value if needed
    const strengthSlider = document.getElementById('txt2imgStrength');
    if (strengthSlider) {
        strengthSlider.value = 0.75;
        if (document.getElementById('txt2imgStrengthValue')) {
            document.getElementById('txt2imgStrengthValue').textContent = '0.75';
        }
    }

    // Optional: Show a small feedback message
    showMessage('Image cleared', 'success');
}

// Open modal
function openModal(src) {
// Hide video if it exists
var videoModal = document.getElementById('modalVideo');
if (videoModal) {
videoModal.style.display = 'none';
videoModal.pause();
}

if (elements.modalImage) {
elements.modalImage.style.display = 'block';
elements.modalImage.src = src;
}
if (elements.imageModal) {
elements.imageModal.classList.add('active');
}
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initializeApp);
