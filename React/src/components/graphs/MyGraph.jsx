// MyGraph.jsx
import { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import PropTypes from 'prop-types';
import { highlightNodeAndEdges } from '../../hooks/HighlightNodeAndEdges';

const MyGraph = ({ updateTrigger }) => {
    const svgRef = useRef(null);
    const containerRef = useRef(null);
    const [graphData, setGraphData] = useState(null);

    // MongoDB에서 데이터를 가져오는 부분
    const loadGraphData = useCallback(async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/articles');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const articles = await response.json();

            const combinedNodes = new Map();
            const combinedEdges = new Map();
            const articleIds = [];

            articles.forEach((article) => {
                const articleId = article._id;
                const hypergraphData = article.hypergraph_data;

                // 노드 처리
                hypergraphData.nodes.forEach(node => {
                    if (!combinedNodes.has(node.id)) {
                        combinedNodes.set(node.id, {
                            id: node.id,
                            importance: node.importance,
                            connections: 0,
                            articles: [articleId]
                        });
                    } else {
                        const existingNode = combinedNodes.get(node.id);
                        existingNode.importance = Math.max(existingNode.importance, node.importance);
                        if (!existingNode.articles.includes(articleId)) {
                            existingNode.articles.push(articleId);
                        }
                    }
                });

                // 엣지 처리
                hypergraphData.edges.forEach(edge => {
                    const [sourceId, targetId] = edge.nodes.sort();
                    const edgeKey = `${sourceId}-${targetId}`;

                    if (!combinedEdges.has(edgeKey)) {
                        combinedEdges.set(edgeKey, {
                            source: sourceId,
                            target: targetId,
                            weight: 1,
                            articles: [articleId],
                            descriptions: [edge.description],
                            categories: [edge.category]
                        });
                    } else {
                        const existingEdge = combinedEdges.get(edgeKey);
                        existingEdge.weight++;
                        if (!existingEdge.articles.includes(articleId)) {
                            existingEdge.articles.push(articleId);
                        }
                        if (edge.category && !existingEdge.categories.includes(edge.category)) {
                            existingEdge.categories.push(edge.category);
                        }
                    }

                    if (combinedNodes.has(sourceId)) {
                        const sourceNode = combinedNodes.get(sourceId);
                        sourceNode.connections++;
                    }
                    if (combinedNodes.has(targetId)) {
                        const targetNode = combinedNodes.get(targetId);
                        targetNode.connections++;
                    }
                });

                articleIds.push(articleId);
            });

            const graphData = {
                nodes: Array.from(combinedNodes.values()),
                edges: Array.from(combinedEdges.values()),
                articles: articleIds
            };

            setGraphData(graphData);
        } catch (error) {
            console.error('Error loading graph data:', error);
        }
    }, []);

    const renderGraph = useCallback(() => {
        if (!graphData || !svgRef.current || !containerRef.current) return;
    
        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();
    
        const container = containerRef.current;
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Create tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('pointer-events', 'none')
            .style('background-color', 'white')
            .style('padding', '5px 10px')
            .style('border', '1px solid #999')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)');
    
        svg.attr('width', width).attr('height', height);
    
        const nodeRadii = graphData.nodes.map(d => {
            const baseRadius = 10 + (d.importance * 50);
            const connectionRadius = d.connections * 3;
            return Math.min(baseRadius + connectionRadius, 50);
        });
    
        const simulation = d3.forceSimulation(graphData.nodes)
            .force('link', d3.forceLink(graphData.edges)
                .id(d => d.id)
                .distance(d => {
                    const sourceRadius = nodeRadii[graphData.nodes.findIndex(n => n.id === d.source.id)];
                    const targetRadius = nodeRadii[graphData.nodes.findIndex(n => n.id === d.target.id)];
                    return Math.max(sourceRadius + targetRadius + 20, 50);
                })
            )
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide()
                .radius(d => {
                    const index = graphData.nodes.indexOf(d);
                    return nodeRadii[index] + 10;
                })
            );
    
        const graphContainer = svg.append("g")
            .attr("class", "graph-container");
    
        const link = graphContainer.append('g')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .selectAll('path')
            .data(graphData.edges.flatMap(edge => {
                return Array.from({ length: edge.weight }, (_, i) => ({
                    ...edge,
                    offset: i - (edge.weight - 1) / 2,
                    index: i
                }));
            }))
            .enter().append('path')
            .attr('fill', 'none')
            .attr('stroke-width', 1)
            .on('mouseover', (event, d) => {
                tooltip.transition()
                    .style('opacity', 0.9);
                tooltip.html(`
                    <strong>${d.source.id || d.source} ↔ ${d.target.id || d.target}</strong><br/>
                    ${d.categories.join('<br/>')}
                `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', () => {
                tooltip.transition()
                    .duration(300)
                    .style('opacity', 0);
            });

        const node = graphContainer.append('g')
            .selectAll('circle')
            .data(graphData.nodes)
            .enter().append('circle')
            .attr('r', (d, i) => nodeRadii[i])
            .attr('fill', d => {
                if (d.connections > 4) return '#32CD32';
                if (d.connections > 2) return '#32CD32';
                return '#999999';
            })
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
                })
            )
            .on('click', (event, d) => {
                console.log('Node clicked:', d);
                const currentSvg = d3.select(svgRef.current);
                highlightNodeAndEdges(graphData, currentSvg, d);
            });

        const labels = graphContainer.append('g')
            .selectAll('text')
            .data(graphData.nodes)
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', 10)
            .attr('dx', 0)
            .attr('dy', 0)
            .attr('fill', '#333')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle');

        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {
                graphContainer.attr("transform", event.transform);
            });

        svg.call(zoom).call(zoom.transform, d3.zoomIdentity);

        simulation.on('tick', () => {
            link.attr('d', d => {
                const sourceX = d.source.x;
                const sourceY = d.source.y;
                const targetX = d.target.x;
                const targetY = d.target.y;

                const spacing = 12;
                const curvature = 0.1 * (1 + d.index * 0.05);
                
                const midX = (sourceX + targetX) / 2;
                const midY = (sourceY + targetY) / 2;
                
                const dx = targetX - sourceX;
                const dy = targetY - sourceY;
                const normalX = -dy;
                const normalY = dx;
                const normalLength = Math.sqrt(normalX * normalX + normalY * normalY);
                
                const offsetX = (normalX / normalLength) * spacing * d.offset;
                const offsetY = (normalY / normalLength) * spacing * d.offset;
                
                const controlX = midX + offsetX;
                const controlY = midY + offsetY;
                
                return `M${sourceX},${sourceY}
                        Q${controlX},${controlY}
                        ${targetX},${targetY}`;
            });

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        return () => {
            tooltip.remove();
        };
    }, [graphData]);

    // updateTrigger가 변경될 때마다 데이터를 다시 로드
    useEffect(() => {
        loadGraphData();
    }, [loadGraphData, updateTrigger]);

    useEffect(() => {
        if (graphData) {
            const cleanup = renderGraph();
            return () => {
                cleanup && cleanup();
                d3.selectAll('.tooltip').remove();
            };
        }

        const handleResize = () => {
            if (graphData) {
                const cleanup = renderGraph();
                cleanup && cleanup();
            }
        };

        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
            d3.selectAll('.tooltip').remove();
        };
    }, [graphData, renderGraph]);

    return (
        <div 
            ref={containerRef} 
            className="network-graph" 
            style={{
                width: '100%', 
                height: '75vh', 
                overflow: 'hidden'
            }}
        >
            <svg ref={svgRef} style={{width: '100%', height: '100%'}}></svg>
        </div>
    );
};

MyGraph.propTypes = {
    updateTrigger: PropTypes.number.isRequired
};

export default MyGraph;