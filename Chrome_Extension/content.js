// content.js
(function() {
    console.log('Content script loaded');

    const currentUrl = window.location.href;
    if (currentUrl.includes('hankyung.com/article') || (/mk\.co\.kr\/news\/[^\/]+\/\d+$/.test(currentUrl)) || currentUrl.includes('n.news.naver.com/mnews/article') || currentUrl.includes('v.daum.net/v')) {
        const buttonContainer = document.createElement('div');
        buttonContainer.style.cssText = `
            position: fixed;
            top: 50%;
            right: 20px;
            transform: translateY(-50%);
            z-index: 9999;
            transition: transform 0.2s ease;
        `;

        const button = document.createElement('button');
        button.textContent = '기사 분석';
        button.style.cssText = `
            padding: 10px; /* 패딩을 조정하여 버튼 크기를 조절 */
            background-color: #32CD32;
            color: white;
            border: none;
            border-radius: 50%; /* 버튼을 원형으로 만들기 */
            cursor: pointer;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            user-select: none;
            transition: all 0.2s ease;
            width: 60px;
            height: 60px;
        `;

        // 호버 효과 추가
        button.addEventListener('mouseover', () => {
            button.style.transform = 'scale(1.1)';
            button.style.boxShadow = '0 0 16px rgba(0,0,0,0.2)';
        });

        button.addEventListener('mouseout', () => {
            button.style.transform = 'scale(1)';
            button.style.boxShadow = 'none';
        });

        buttonContainer.appendChild(button);

        // 버튼 클릭 이벤트
        button.onclick = (e) => {    
            chrome.runtime.sendMessage({
                action: "openPanel",
                type: "url",
                url: currentUrl
            });
        };
        
        document.body.appendChild(buttonContainer);
    }
})();