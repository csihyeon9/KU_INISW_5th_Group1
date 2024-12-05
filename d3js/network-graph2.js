
    const svg = d3.select('#network-svg');
    const width = +svg.attr('width');
    const height = +svg.attr('height');

    let graphData = null;
    let activeArticle = null;

    // Load JSON files from ./data folder
    async function loadGraphData() {
        try {
            // Fetch JSON files from ./data directory
            const response = await fetch('./data');
            const text = await response.text();
            
            // Use a simple DOM parser to extract file names
            const parser = new DOMParser();
            const htmlDoc = parser.parseFromString(text, 'text/html');
            const jsonFiles = Array.from(htmlDoc.querySelectorAll('a'))
                .map(a => a.href)
                .filter(href => href.endsWith('.json'))
                .map(href => href.split('/').pop());

            const combinedNodes = new Map();
            const combinedEdges = [];
            const articles = [];

            // Load each JSON file
            for (const fileName of jsonFiles) {
                const fileResponse = await fetch(`./data/${fileName}`);
                const jsonData = await fileResponse.json();

                // Combine nodes
                jsonData.nodes.forEach(node => {
                    if (!combinedNodes.has(node.id)) {
                        combinedNodes.set(node.id, { 
                            id: node.id, 
                            importance: node.importance,
                            articles: [fileName]
                        });
                    } else {
                        const existingNode = combinedNodes.get(node.id);
                        existingNode.importance = Math.max(existingNode.importance, node.importance);
                        if (!existingNode.articles.includes(fileName)) {
                            existingNode.articles.push(fileName);
                        }
                    }
                });

                // Combine edges
                jsonData.edges.forEach(edge => {
                    combinedEdges.push({
                        ...edge,
                        source: edge.nodes[0],
                        target: edge.nodes[1],
                        article: fileName
                    });
                });

                articles.push(fileName);
            }

            // Prepare graph data
            graphData = {
                nodes: Array.from(combinedNodes.values()),
                edges: combinedEdges,
                articles: articles
            };

            // Render graph and create article tabs
            renderGraph();
            createArticleTabs();
        } catch (error) {
            console.error('Error loading graph data:', error);
            alert('Failed to load JSON files. Make sure ./data folder exists and contains JSON files.');
        }
    }

    // 포스 시뮬레이션 이후에 줌과 드래그 설정
    function setupZoomAndDrag(simulation) {
        // 모든 그래프 요소를 포함할 컨테이너 그룹 생성
        const g = svg.append("g")
            .attr("class", "graph-container");

        // 줌 비헤이비어 설정
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])  // 줌 범위 조정
            .on("zoom", (event) => {
                // 그래프 컨테이너에 직접 변환 적용
                g.attr("transform", event.transform);
            });

        // SVG에 줌 이벤트 바인딩
        svg.call(zoom)
        .call(zoom.transform, d3.zoomIdentity);  // 초기 변환 설정

        // 드래그 비헤이비어 설정 (노드용)
        const drag = d3.drag()
            .on("start", (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
                
                // 드래그 시작 시 이벤트 전파 중지
                event.sourceEvent.stopPropagation();
            })
            .on("drag", (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on("end", (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });

        // 노드에 드래그 적용
        node.call(drag);
    }

    // Render network graph
    function renderGraph() {
        // Clear previous graph
        svg.selectAll('*').remove();

        const width = window.innerWidth;
        const height = window.innerHeight;

        svg.attr('width', width).attr('height', height);

        // Create force simulation with improved containment
        const simulation = d3.forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.edges).id(d => d.id))
        .force('charge', d3.forceManyBody().strength(-10))  // Slightly stronger repulsion for full screen
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide(20))  // Larger collision radius
        // Add boundary force to keep nodes within SVG
        .force('boundary', () => {
            graphData.nodes.forEach(node => {
                const padding = 50; // Increased padding for full screen
                node.x = Math.max(padding, Math.min(width - padding, node.x));
                node.y = Math.max(padding, Math.min(height - padding, node.y));
            });
        });

        // Add window resize event listener
        window.addEventListener('resize', () => {
            // Resize SVG to new window dimensions
            svg.attr('width', window.innerWidth)
            .attr('height', window.innerHeight);
            
            // Re-render the graph
            if (graphData) {
                renderGraph();
            }
        });


        // Create edges
        const link = svg.append('g')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .selectAll('line')
            .data(graphData.edges)
            .enter().append('line')
            .attr('stroke-width', 2)
            .style('opacity', d => activeArticle ? d.article === activeArticle ? 1 : 0.1 : 1);

        // Create nodes
        const node = svg.append('g')
            .selectAll('circle')
            .data(graphData.nodes)
            .enter().append('circle')
            .attr('r', d => 5 + (d.importance * 10))
            .attr('fill', d => {
                if (d.articles.length > 5) {
                    return '#6A8734'; // For highly connected nodes
                } else if (d.articles.length > 2) {
                    return '#698532'; // For moderately connected nodes
                } else {
                    return '#555555'; // For less connected nodes
                }
            })
            .style('opacity', d => 
                activeArticle 
                    ? d.articles.includes(activeArticle) 
                        ? 1 
                        : 0.1 
                    : 1
            )
            .call(d3.drag()
                .on('start', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on('drag', (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on('end', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }));

        // Add labels to nodes
        const labels = svg.append('g')
            .selectAll('text')
            .data(graphData.nodes)
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', 10)
            .attr('dx', 12)
            .attr('dy', 4)
            .style('opacity', d => 
                activeArticle 
                    ? d.articles.includes(activeArticle) 
                        ? 1 
                        : 0.1 
                    : 1
            );

        // Update positions on each tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }
    // let z = d3.zoomIdentity;
    

        const g = svg.append("g")
        .attr("cursor", "grab")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));
        
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y; 

            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
            }
            
        
        const zoom = d3.zoom()
            .extent([[0, 0], [width, height]])
            .scaleExtent([1, 8])
            .on('zoom', (event) => {
                svg.attr('transform', event.transform);
            });

        function zoomed({transform}) {
                g.attr("transform", transform);
            }   

        // // 줌 동작을 SVG에 바인딩
        svg.call(zoom)
            .call(zoom.transform, d3.zoomIdentity); 
    

    // Create article tabs
    function createArticleTabs() {
        const articleList = d3.select('#article-list');
        
        // Clear existing elements
        articleList.selectAll('*').remove();
        
        // Create search input inside article-list
        // const searchContainer = articleList.append('div')
        //     .attr('class', 'w-full mb-2');
    
        const searchInput = articleList.append('input')
            .attr('type', 'text')
            .attr('placeholder', 'Search articles...')
            .attr('class', 'search w-full mb-2 px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300')
            
        // Create buttons for each article
        const articleButtons = articleList.selectAll('button.article-btn')
            .data(graphData.articles)
            .enter()
            .append('button')
            .attr('class', ' w-full mb-2 px-4 py-2 bg-gray-200 text-gray-800 rounded hover:bg-gray-300')
            .text(d => d)
            .on('click', (event, article) => {
                activeArticle = article;
                renderGraph();
            });
        
        // View All button
        articleList.append('button')
            .text('View All')
            .attr('class', 'w-full mb-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600')
            .on('click', () => {
                activeArticle = null;
                renderGraph();
            });
        
        // Add event listener for search input
        searchInput.on('input', function() {
            const searchTerm = this.value.toLowerCase();
            
            // Hide/show buttons based on search term
            articleList.selectAll('button.article-btn')
                .style('display', function(d) {
                    const articleText = d.toLowerCase();
                    return (searchTerm === '' || articleText.includes(searchTerm)) ? 'block' : 'none';
                });
        });

    }

      

    // Initial load

    loadGraphData();
