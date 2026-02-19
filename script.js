// ============================================================================
// CONFIGURACIÓN GLOBAL
// ============================================================================
const IMG_SIZE = 64; // imagenes de 64x64
const CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'];
const MODEL_PATH = 'web_model/model.json';

let model = null;

// ============================================================================
// REFERENCIAS UI
// ============================================================================
const imgEl = document.getElementById('previewImage');
const placeholderEl = document.getElementById('placeholder');
const resultEl = document.getElementById('predictionResult');
const confidenceEl = document.getElementById('confidenceScore');
const statusEl = document.getElementById('modelStatus');
const predictBtn = document.getElementById('predictBtn');
const manualUploadSection = document.getElementById('manualUploadSection');

function logDebug(msg, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${msg}`);
}

// ============================================================================
// INICIO
// ============================================================================
window.addEventListener('DOMContentLoaded', async () => {
    if (window.location.protocol === 'file:') {
        showManualUpload();
    } else {
        await findAndLoadModel();
    }
});

// ============================================================================
// LÓGICA DE CARGA Y PARCHEO
// ============================================================================
async function findAndLoadModel() {
    logDebug("Iniciando carga automática...", 'info');
    const cacheBuster = `?t=${Date.now()}`;

    try {
        const modelUrl = new URL(MODEL_PATH, window.location.href).href;

        // 1. Descargar JSON
        const response = await fetch(modelUrl + cacheBuster);
        if (!response.ok) throw new Error("JSON no encontrado");

        const jsonContent = await response.text();
        const modelData = JSON.parse(jsonContent);

        // 2. APLICAR PARCHES (Input Shape y Nombres de Pesos)
        patchModelArchitecture(modelData);

        // 3. Descargar BINs
        const basePath = modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1);
        const binFiles = [];

        if (modelData.weightsManifest) {
            for (const group of modelData.weightsManifest) {
                for (const p of group.paths) {
                    const cleanPath = p.startsWith('./') ? p.slice(2) : p;
                    const binUrl = basePath + cleanPath;

                    const binResp = await fetch(binUrl + cacheBuster);
                    if (!binResp.ok) throw new Error(`Binario faltante: ${cleanPath}`);

                    const blob = await binResp.blob();
                    binFiles.push(new File([blob], cleanPath, { type: 'application/octet-stream' }));
                }
            }
        }

        // 4. Cargar modelo parcheado
        const jsonFile = new File([JSON.stringify(modelData)], 'model.json', { type: 'application/json' });
        model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, ...binFiles]));

        logDebug(`🎉 MODELO CARGADO EXITOSAMENTE`, 'success');
        updateStatusSuccess();

    } catch (e) {
        logDebug(`Fallo carga automática: ${e.message}`, 'error');
        showManualUpload();
    }
}

// === FUNCIÓN CRÍTICA DE CORRECCIÓN ===
function patchModelArchitecture(modelData) {
    // Corregir batch_shape
    const layers = modelData.modelTopology?.model_config?.config?.layers;
    if (layers) {
        layers.forEach(layer => {
            if (layer.class_name === 'InputLayer' && layer.config) {
                if (layer.config.batch_shape && !layer.config.batchInputShape) {
                    console.log("🔧 Parche: batchInputShape agregado");
                    layer.config.batchInputShape = layer.config.batch_shape;
                }
            }
        });
    }

    // Corregir Nombres de Pesos
    const modelName = modelData.modelTopology?.model_config?.config?.name;
    if (modelName && modelData.weightsManifest) {
        const prefix = modelName + '/';
        console.log(`🔧 Parche: Buscando prefijo '${prefix}' en pesos...`);

        let fixedCount = 0;
        modelData.weightsManifest.forEach(group => {
            group.weights.forEach(w => {
                if (w.name.startsWith(prefix)) {
                    w.name = w.name.slice(prefix.length);
                    fixedCount++;
                }
            });
        });
        console.log(`🔧 Parche: Se renombraron ${fixedCount} pesos.`);
    }
}

// ============================================================================
// UI & PREDICT
// ============================================================================
function updateStatusSuccess() {
    statusEl.innerHTML = `
        <div class="flex items-center gap-2 text-emerald-400">
            <i data-lucide="check-circle" class="w-4 h-4"></i>
            <span>Modelo listo</span>
        </div>`;
    predictBtn.disabled = false;
    predictBtn.classList.replace('bg-slate-600', 'bg-blue-600');
    predictBtn.classList.replace('text-slate-400', 'text-white');
    predictBtn.classList.replace('cursor-not-allowed', 'hover:bg-blue-500');
    predictBtn.innerText = "DETECTAR SIGNO";
    if (window.lucide) lucide.createIcons();
}

function showManualUpload() {
    statusEl.innerHTML = `
        <div class="flex items-center gap-2 text-yellow-400 font-bold">
            <i data-lucide="alert-circle" class="w-4 h-4"></i>
            <span>Esperando archivos...</span>
        </div>`;
    manualUploadSection.classList.remove('hidden');
    if (window.lucide) lucide.createIcons();
    predictBtn.innerText = "Carga manual requerida";
}

window.handleImageUpload = function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imgEl.src = e.target.result;
            imgEl.classList.remove('hidden');
            placeholderEl.classList.add('hidden');
            resultEl.innerText = "--";
            confidenceEl.style.opacity = "0";
            if (model) predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

window.predict = async function () {
    if (imgEl.classList.contains('hidden')) return alert("Sube una imagen.");
    if (!model) return alert("Modelo no cargado.");

    predictBtn.innerText = "Analizando...";

    setTimeout(() => {
        tf.tidy(() => {
            let tensor = tf.browser.fromPixels(imgEl);
            tensor = tf.image.resizeBilinear(tensor, [IMG_SIZE, IMG_SIZE]);
            tensor = tensor.mean(2); // RGB -> Grayscale
            tensor = tensor.expandDims(-1); // [64,64,1]
            tensor = tensor.div(255.0);
            tensor = tensor.expandDims(0);

            const predictions = model.predict(tensor);
            const data = predictions.dataSync();
            const maxIndex = predictions.argMax(-1).dataSync()[0];

            updateUI(CLASSES[maxIndex] || "?", data[maxIndex]);
        });
        predictBtn.innerText = "DETECTAR SIGNO";
    }, 50);
}

function updateUI(label, confidence) {
    resultEl.innerText = label;
    confidenceEl.innerText = `Confianza: ${(confidence * 100).toFixed(1)}%`;
    confidenceEl.style.opacity = "1";
    confidenceEl.className = confidence > 0.80
        ? "mt-2 font-mono text-lg text-emerald-400 font-bold"
        : "mt-2 font-mono text-lg text-yellow-400";
}

// ============================================================================
// TOGGLE MODO EDUCATIVO
// ============================================================================
window.toggleMode = function () {
    const body = document.getElementById('appBody');
    const btnText = document.getElementById('modeText');
    const btnIcon = document.getElementById('modeIcon');

    // Alternar clase
    body.classList.toggle('edu-mode');

    // Cambiar texto del botón
    if (body.classList.contains('edu-mode')) {
        btnText.innerText = "Volver al Modo Tech";
        btnIcon.innerText = "💻";
        // Sincronizar el mensaje de estado simple
        const statusText = document.getElementById('modelStatus').innerText;
        document.getElementById('simpleStatus').innerText = statusText.includes("listo") ? "Modelo Listo" : "Cargando...";
    } else {
        btnText.innerText = "Modo Educativo";
        btnIcon.innerText = "🎓";
    }
}

// Hook para actualizar el estado simple en modo educativo cuando el modelo cargue
const originalUpdateStatus = updateStatusSuccess;
updateStatusSuccess = function () {
    originalUpdateStatus(); // Ejecutar la original
    const simple = document.getElementById('simpleStatus');
    if (simple) simple.innerText = "¡Listo para usar!";
}

// NUEVO: Función para cargar imágenes predefinidas desde el desplegable
window.loadPredefinedImage = function (imagePath) {
    if (!imagePath) return; // Si vuelve a "-- Elige una imagen --", no hace nada

    // Actualizar la fuente de la imagen con la ruta seleccionada
    imgEl.src = imagePath;

    // Mostrar la imagen y ocultar el placeholder
    imgEl.classList.remove('hidden');
    placeholderEl.classList.add('hidden');

    // Resetear el resultado y la confianza
    resultEl.innerText = "--";
    confidenceEl.style.opacity = "0";

    // Habilitar el botón de predicción si el modelo ya está cargado
    if (model) {
        predictBtn.disabled = false;
        predictBtn.classList.replace('bg-slate-600', 'bg-blue-600');
        predictBtn.classList.replace('text-slate-400', 'text-white');
        predictBtn.classList.replace('cursor-not-allowed', 'hover:bg-blue-500');
    }
}

// ============================================================================
// LÓGICA DEL BOTÓN DIAL INFERIOR
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    const primaryBtn = document.getElementById('dialMainBtn');
    const dialContainer = document.querySelector('.dial-btn__container');
    let isDialActive = false;

    // Inicializar iconos de Lucide (si no están cargados ya)
    if (window.lucide) window.lucide.createIcons();

    const closeDial = () => {
        primaryBtn.classList.remove('dial-btn--active');
        isDialActive = false;
        document.removeEventListener('click', handleOutsideClickDial);
    };

    const handleOutsideClickDial = (e) => {
        if (!dialContainer.contains(e.target)) {
            closeDial();
        }
    };

    primaryBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!isDialActive) {
            primaryBtn.classList.add('dial-btn--active');
            isDialActive = true;
            document.addEventListener('click', handleOutsideClickDial);
        } else {
            closeDial();
        }
    });

    // Cerrar el dial si se hace clic en alguna de las opciones
    const optionBtns = document.querySelectorAll('.dial-btn--option');
    optionBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            closeDial();
        });
    });
});