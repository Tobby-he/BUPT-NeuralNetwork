document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const description = document.getElementById('description');
    const errorMsg = document.getElementById('errorMessage');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // 处理文件上传
    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                description.textContent = '';
                errorMsg.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    }

    // 点击上传区域触发文件选择
    dropZone.addEventListener('click', () => imageInput.click());

    // 文件选择处理
    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // 拖放处理
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        handleFile(e.dataTransfer.files[0]);
    });

    // 处理粘贴事件
    document.addEventListener('paste', (e) => {
        const items = e.clipboardData.items;
        for (let item of items) {
            if (item.type.startsWith('image/')) {
                handleFile(item.getAsFile());
                break;
            }
        }
    });
});

async function generateDescription() {
    const imageInput = document.getElementById('imageInput');
    const modelSelect = document.getElementById('modelSelect');
    const description = document.getElementById('description');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMsg = document.getElementById('errorMessage');

    if (!imageInput.files || !imageInput.files[0]) {
        errorMsg.textContent = '请先选择一张图片';
        errorMsg.style.display = 'block';
        return;
    }

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    formData.append('model', modelSelect.value);

    try {
        description.textContent = '';
        errorMsg.style.display = 'none';
        loadingIndicator.style.display = 'block';

        const response = await fetch('http://127.0.0.1:5000/generate', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        description.textContent = data.description || '无法生成描述';
    } catch (error) {
        console.error('Error:', error);
        errorMsg.textContent = `生成描述时出错: ${error.message}`;
        errorMsg.style.display = 'block';
    } finally {
        loadingIndicator.style.display = 'none';
    }
}
 
