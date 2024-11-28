// content.js
(function() {
    console.log('Content script loaded');
    let dragStartX = 0;
    let iconWasClicked = false;
    let lastSelectedText = '';  // 마지막으로 선택된 텍스트를 저장

    document.addEventListener('mousedown', (e) => {
        dragStartX = e.clientX;
        const icon = document.getElementById('finwise-icon');
        
        if (icon && !icon.contains(e.target)) {
            icon.remove();
            // 아이콘이 클릭된 상태에서 동일한 텍스트를 클릭한 경우가 아니라면 선택 상태 초기화
            if (!(iconWasClicked && window.getSelection().toString().trim() === lastSelectedText)) {
                window.getSelection().removeAllRanges();
            }
        }
    });

    document.addEventListener('mouseup', function(e) {
        const selectedText = window.getSelection().toString().trim();
        
        // 아이콘이 클릭된 상태에서 동일한 텍스트를 선택한 경우 리턴
        if (iconWasClicked && selectedText === lastSelectedText) {
            return;
        }

        // 새로운 텍스트가 선택되면 상태 초기화
        if (selectedText !== lastSelectedText) {
            iconWasClicked = false;
        }

        // 기존 아이콘 제거
        const oldIcon = document.getElementById('finwise-icon');
        if (oldIcon) {
            oldIcon.remove();
        }

        if (selectedText) {
            // 텍스트 노드인지 확인
            const selection = window.getSelection();
            const range = selection.getRangeAt(0);
            const parentElement = range.commonAncestorContainer;
            
            // 이미지나 다른 요소를 선택했을 경우 아이콘을 표시하지 않음
            if (parentElement.nodeType !== Node.TEXT_NODE && 
                !parentElement.innerText && 
                !parentElement.textContent) {
                return;
            }
            
            console.log('Text selected:', selectedText);
            
            // 아이콘 생성
            const icon = document.createElement('div');
            icon.id = 'finwise-icon';

            const img = document.createElement('img');
            img.src = chrome.runtime.getURL('images/icon.png');
            img.style.width = '100%';
            img.style.height = '100%';
            img.style.objectFit = 'contain';
            icon.appendChild(img);
            
            // 기본 스타일
            icon.style.cssText = `
                position: fixed;
                width: 20px;
                height: 20px;
                background: white;
                border-radius: 50%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                z-index: 9999999;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                user-select: none;
            `;
            
            // 마우스 드래그가 끝난 위치를 기준으로 아이콘 위치 설정
            const isLeftToRight = dragStartX < e.clientX;

            if (isLeftToRight) {
                icon.style.left = `${e.clientX + 5}px`;
                icon.style.top = `${e.clientY + 5}px`;
            } else {
                icon.style.left = `${e.clientX - 35}px`;
                icon.style.top = `${e.clientY - 35}px`;
            }
            
            // 클릭 이벤트 추가
            icon.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                iconWasClicked = true;
                lastSelectedText = selectedText;  // 마지막으로 선택된 텍스트 저장
                console.log('Icon clicked, selected text:', selectedText);
                
                chrome.runtime.sendMessage({
                    action: "openPanel",
                    text: selectedText
                }, function(response) {
                    console.log('Response received:', response);
                });
                
                icon.remove();
            };
            
            // 마우스 이벤트 중지
            icon.onmousedown = function(e) {
                e.preventDefault();
                e.stopPropagation();
            };
            
            icon.onmouseup = function(e) {
                e.preventDefault();
                e.stopPropagation();
            };
            
            document.body.appendChild(icon);
        }
    });
})();