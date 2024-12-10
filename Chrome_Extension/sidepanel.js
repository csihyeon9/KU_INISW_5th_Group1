// sidepanel.js
let currentResizeObserver = null;

document.addEventListener('DOMContentLoaded', () => {
    const contentDiv = document.getElementById('content');
    const loadingDiv = document.getElementById('loading');
    const saveButton = document.getElementById('saveButton');

    let initialMessage = '기사 분석을 시작해보세요.';

    // 헤더 버튼 이벤트 리스너
    document.getElementById('analysisButton').addEventListener('click', function() {
        this.classList.add('active');
        document.getElementById('personalButton').classList.remove('active');
    });

    document.getElementById('personalButton').addEventListener('click', function() {
        chrome.tabs.create({ url: 'http://localhost:5173' });
    });

    // 저장 버튼 클릭 이벤트
    saveButton.addEventListener('click', async () => {
        console.log('Save button clicked');  // 버튼 클릭 확인
        
        // chrome storage 저장 로직
        chrome.storage.local.get(['graphData', 'savedAnalyses'], function(result) {
            console.log('Chrome storage data:', result);
            if (result.graphData) {
                const currentData = {
                    timestamp: new Date().toISOString(),
                    content: contentDiv.innerHTML,
                    graphData: result.graphData,
                    type: 'permanent'
                };
                
                let savedAnalyses = result.savedAnalyses || [];
                savedAnalyses.push(currentData);
                
                chrome.storage.local.set({ savedAnalyses: savedAnalyses }, function() {
                    saveButton.disabled = true;
                });
            }
        });

        // MongoDB 저장 부분
        chrome.storage.local.get(['originalData'], async function(result) {
            console.log('Attempting MongoDB save with data:', result.originalData);
            if (result.originalData) {
                try {
                    console.log('Making fetch request to save_article');
                    // const response = await fetch('https://finwise.p-e.kr:8000/save_article', {
                    const response = await fetch('http://127.0.0.1:8000/save_article', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(result.originalData)
                    });
                    console.log('Save article response:', response);

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const responseData = await response.json();
                    console.log('Save article response data:', responseData);
                    alert('분석 결과가 성공적으로 저장되었습니다!');
                } catch (error) {
                    console.error('Detailed save error:', error);
                    alert('DB 저장 중 오류가 발생했습니다: ' + error.message);
                }
            } else {
                console.log('No original data found for MongoDB save');
            }
        });
    });

    resetButton.addEventListener('click', () => {
        chrome.storage.local.remove(['lastAnalysis', 'graphData', 'articleTitle'], function() {
            contentDiv.innerHTML = initialMessage;
            saveButton.style.display = 'none';
            document.getElementById('articleTitle').style.display = 'none';  // 제목 숨기기
            document.querySelector('.button-group').style.display = 'none';  // 버튼 그룹 숨기기
        });
    });

    // 초기 상태 로드
    chrome.storage.local.get(['lastAnalysis', 'graphData', 'savedAnalyses', 'articleTitle'], function(result) {
        if (result.lastAnalysis && result.graphData) {
            contentDiv.innerHTML = result.lastAnalysis;

            const titleDiv = document.getElementById('articleTitle');
            if (result.articleTitle) {
                titleDiv.textContent = result.articleTitle;
                titleDiv.style.display = 'block';
                document.querySelector('.button-group').style.display = 'flex';
            }
            
            const isDuplicate = result.savedAnalyses?.some(analysis => 
                JSON.stringify(analysis.graphData) === JSON.stringify(result.graphData)
            );

            saveButton.style.display = 'block';
            if (isDuplicate) {
                // saveButton.textContent = '저장됨';
                saveButton.disabled = true;
            } else {
                // saveButton.textContent = '저장';
                saveButton.disabled = false;
            }

            const graphContainer = document.getElementById('graph-container');
            if (graphContainer) {
                if (currentResizeObserver) {
                    currentResizeObserver.disconnect();
                }
                
                currentResizeObserver = new ResizeObserver(entries => {
                    for (let entry of entries) {
                        if (entry.target.id === 'graph-container') {
                            visualizeHypergraph(result.graphData);
                        }
                    }
                });
                currentResizeObserver.observe(graphContainer);
    
                requestAnimationFrame(() => {
                    visualizeHypergraph(result.graphData);
                });
            }
        } else {
            contentDiv.innerHTML = initialMessage;
        }
    });

    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "analyzeText") {
            const analysisData = request.type === "url"
                ? { url: request.url }
                : { text: request.text };

            handleAnalysis(analysisData);
            sendResponse({ received: true });
            return false;
        }
    });

    async function handleAnalysis(analysisData) {
        try {
            contentDiv.classList.add('loading-active');
            loadingDiv.style.display = 'block';
            contentDiv.innerHTML = '';
            document.getElementById('articleTitle').style.display = 'none';  // 제목 숨기기
            document.querySelector('.button-group').style.display = 'none';  // 버튼 그룹 숨기기

            // const response = await fetch('https://finwise.p-e.kr:8000/service', {
            const response = await fetch('http://127.0.0.1:8000/service', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(analysisData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received data:', data);

            const titleDiv = document.getElementById('articleTitle');
            if (data.title) {
                titleDiv.textContent = data.title;
                titleDiv.style.display = 'block';
                document.querySelector('.button-group').style.display = 'flex';
            } else {
                titleDiv.style.display = 'none';
            }

            if (currentResizeObserver) {
                currentResizeObserver.disconnect();
            }

            currentResizeObserver = new ResizeObserver(entries => {
                for (let entry of entries) {
                    if (entry.target.id === 'graph-container') {
                        visualizeHypergraph(data.hypergraph_data);
                    }
                }
            });
            
            const formatDate = (dateStr) => {
                const cleanedDateStr = dateStr
                    .replace('T', ' ')
                    .split('.')[0];
                return cleanedDateStr;
            };
            
            const analysisHTML = `
                <div id="graph-container">
                    <svg></svg>
                </div>
                <div class="definitions-section">
                    <h3>경제 용어를 확인해보세요.</h3>
                    ${Object.entries(data.definitions).map(([term, definition]) => {
                        const maxLength = 200;
                        const needsTruncation = definition && definition.length > maxLength;
                        const truncatedText = needsTruncation ? 
                            definition.slice(0, maxLength) + '...' : 
                            definition;

                            return `
                            <div class="definition-item" data-full-text="${definition ? definition.replace(/"/g, '&quot;') : ''}">
                                <div class="definition-header">
                                    <div class="definition-term">${term}</div>
                                    ${needsTruncation ? `
                                        <button class="toggle-definition-btn">
                                            <img src="images/down.png" alt="더 보기" class="toggle-icon">
                                        </button>
                                    ` : ''}
                                </div>
                                <div class="definition-description">
                                    <div class="definition-content">
                                        ${truncatedText ? 
                                            truncatedText
                                                .replace(/\\r\\n/g, '<br>')
                                                .replace(/\r\n/g, '<br>')
                                                .replace(/\\n/g, '<br>')
                                                .replace(/\n/g, '<br>') 
                                            : '정의를 찾을 수 없습니다.'
                                        }
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;

            const addDefinitionToggleListeners = () => {
                const processBreaks = (text) => text
                    .replace(/\\r\\n/g, '<br>')
                    .replace(/\r\n/g, '<br>')
                    .replace(/\\n/g, '<br>')
                    .replace(/\n/g, '<br>');
            
                document.querySelectorAll('.toggle-definition-btn').forEach(button => {
                    button.addEventListener('click', function() {
                        const definitionItem = this.closest('.definition-item');
                        const content = definitionItem.querySelector('.definition-content');
                        const fullText = definitionItem.dataset.fullText;
                        const img = this.querySelector('img');
                        
                        if (img.alt === '더 보기') {
                            content.innerHTML = processBreaks(fullText);
                            img.src = 'images/up.png';
                            img.alt = '접기';
                        } else {
                            content.innerHTML = processBreaks(fullText.slice(0, 200) + '...');
                            img.src = 'images/down.png';
                            img.alt = '더 보기';
                        }
                    });
                });
            };
            
            contentDiv.innerHTML = analysisHTML;
            addDefinitionToggleListeners();

            const graphContainer = document.getElementById('graph-container');
            currentResizeObserver.observe(graphContainer);

            requestAnimationFrame(() => {
                visualizeHypergraph(data.hypergraph_data);
            });

            chrome.storage.local.set({
                lastAnalysis: analysisHTML,
                graphData: data.hypergraph_data,
                articleTitle: data.title,
                originalData: data
            });

            // 새 분석 결과에 대한 저장 버튼 표시
            saveButton.style.display = 'block';
            chrome.storage.local.get(['savedAnalyses'], function(result) {
                const isDuplicate = result.savedAnalyses?.some(analysis => 
                    JSON.stringify(analysis.graphData) === JSON.stringify(data.hypergraph_data)
                );

                if (isDuplicate) {
                    saveButton.disabled = true;
                } else {
                    saveButton.disabled = false;
                }
            });

        } catch (error) {
            document.getElementById('articleTitle').style.display = 'none';  // 에러시 제목 숨기기
            document.querySelector('.button-group').style.display = 'none';  // 에러시 버튼 그룹 숨기기
            saveButton.style.display = 'none';
            const errorHTML = `<div class="error">분석 중 오류가 발생했습니다: ${error.message}</div>`;
            contentDiv.innerHTML = errorHTML;
            chrome.storage.local.set({
                lastAnalysis: errorHTML
            });
            console.error('Analysis error:', error);
        } finally {
            loadingDiv.style.display = 'none';
            contentDiv.classList.remove('loading-active');
        }
    }
});