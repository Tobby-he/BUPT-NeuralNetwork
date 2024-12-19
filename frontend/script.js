document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const generateBtn = document.getElementById('generate-btn');
    const description = document.getElementById('description');
    let currentImage = null;

    // 处理文件上传
    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                imagePreview.innerHTML = '';
                imagePreview.appendChild(img);
                currentImage = file;
            };
            reader.readAsDataURL(file);
        }
    }

    // 点击上传区域触发文件选择
    dropZone.addEventListener('click', () => fileInput.click());

    // 文件选择处理
    fileInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    // 拖放处理
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#666';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#ccc';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#ccc';
        handleFile(e.dataTransfer.files[0]);
    });

    // 处理粘贴事件
    document.addEventListener('paste', (e) => {
        const items = e.clipboardData.items;
        for (let item of items) {
            if (item.type.startsWith('image/')) {
                const file = item.getAsFile();
                handleFile(file);
                break;
            }
        }
    });

    // 生成描述按钮点击事件
    generateBtn.addEventListener('click', async () => {
        if (!currentImage) {
            alert('请先上传图片');
            return;
        }

        try {
            description.textContent = '正在生成描述...';
            
            // 创建 FormData 对象
            const formData = new FormData();
            formData.append('image', currentImage);

            // 这里替换成你的后端 API 地址
            const response = await fetch('YOUR_API_ENDPOINT', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            description.textContent = result.description;
        } catch (error) {
            description.textContent = '生成描述时出错，请重试';
            console.error('Error:', error);
        }
    });
}); 