import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

from config import settings
from vector_store import vector_store_service
from rag_agents import rag_agents
from models import QueryRequest

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_vector_store():
    """Debug vector store dan collection."""
    print("=" * 50)
    print("🔍 DEBUGGING VECTOR STORE")
    print("=" * 50)
    
    try:
        # 1. Check vector store initialization
        print("\n1. Checking vector store initialization...")
        collection_info = await vector_store_service.get_collection_info()
        print(f"   Collection Status: {collection_info.get('status', 'Unknown')}")
        print(f"   Document Count: {collection_info.get('document_count', 0)}")
        print(f"   Collection Name: {collection_info.get('collection_name', 'Unknown')}")
        
        # 2. Check if collection has documents
        document_count = collection_info.get('document_count', 0)
        if document_count == 0:
            print("   ❌ NO DOCUMENTS FOUND IN COLLECTION!")
            print("   💡 Solution: Upload documents first")
            return False
        else:
            print(f"   ✅ Found {document_count} documents in collection")
        
        # 3. Test search functionality
        print("\n2. Testing search functionality...")
        test_queries = [
            "UUD 1945 Pasal 32",
            "Undang-Undang Dasar 1945",
            "kebudayaan nasional"
        ]
        
        for query in test_queries:
            print(f"\n   Testing query: '{query}'")
            try:
                # Try different search methods
                search_methods = []
                
                # Check available search methods
                if hasattr(vector_store_service, 'hybrid_search'):
                    search_methods.append(('hybrid_search', vector_store_service.hybrid_search))
                if hasattr(vector_store_service, 'semantic_search'):
                    search_methods.append(('semantic_search', vector_store_service.semantic_search))
                if hasattr(vector_store_service, 'similarity_search'):
                    search_methods.append(('similarity_search', vector_store_service.similarity_search))
                if hasattr(vector_store_service, 'search'):
                    search_methods.append(('search', vector_store_service.search))
                
                print(f"   Available search methods: {[name for name, _ in search_methods]}")
                
                for method_name, method in search_methods:
                    try:
                        print(f"   Testing {method_name}...")
                        
                        # Call search method
                        if asyncio.iscoroutinefunction(method):
                            results = await method(query, k=3)
                        else:
                            results = method(query, k=3)
                        
                        print(f"   ✅ {method_name}: Found {len(results)} results")
                        
                        # Show first result details
                        if results:
                            first_result = results[0]
                            content_preview = ""
                            metadata = {}
                            
                            # Handle different result formats
                            if hasattr(first_result, 'content'):
                                content_preview = first_result.content[:100]
                                metadata = getattr(first_result, 'metadata', {})
                            elif hasattr(first_result, 'page_content'):
                                content_preview = first_result.page_content[:100]
                                metadata = getattr(first_result, 'metadata', {})
                            elif isinstance(first_result, dict):
                                content_preview = first_result.get('content', first_result.get('page_content', ''))[:100]
                                metadata = first_result.get('metadata', {})
                            
                            print(f"   📄 First result preview: {content_preview}...")
                            print(f"   📋 Metadata: {metadata}")
                        break
                    except Exception as e:
                        print(f"   ❌ {method_name} failed: {e}")
                        continue
                else:
                    print("   ❌ All search methods failed!")
                    
            except Exception as e:
                print(f"   ❌ Search test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store debug failed: {e}")
        return False

async def debug_rag_agents():
    """Debug RAG agents workflow."""
    print("\n" + "=" * 50)
    print("🤖 DEBUGGING RAG AGENTS")
    print("=" * 50)
    
    try:
        # Test query
        test_query = "Ada berapa ayat dari UUD 1945 Pasal 32?"
        print(f"\nTesting query: '{test_query}'")
        
        # Create query request
        query_request = QueryRequest(
            query=test_query,
            include_sources=True,
            max_results=5
        )
        
        print("\n1. Testing standard query processing...")
        try:
            start_time = datetime.now()
            response = await rag_agents.process_query(query_request)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            print(f"   ✅ Query processed in {processing_time:.2f}s")
            print(f"   📝 Answer: {response.answer[:200]}...")
            print(f"   📊 Confidence: {response.confidence_score}")
            print(f"   📚 Sources found: {len(response.sources)}")
            
            # Show sources details
            if response.sources:
                print("\n   📋 Sources details:")
                for i, source in enumerate(response.sources[:3]):
                    print(f"   Source {i+1}:")
                    if hasattr(source, 'content'):
                        print(f"     Content: {source.content[:100]}...")
                        print(f"     Metadata: {getattr(source, 'metadata', {})}")
                    elif hasattr(source, 'page_content'):
                        print(f"     Content: {source.page_content[:100]}...")
                        print(f"     Metadata: {getattr(source, 'metadata', {})}")
                    elif isinstance(source, dict):
                        content = source.get('content', source.get('page_content', 'No content'))
                        print(f"     Content: {content[:100]}...")
                        print(f"     Metadata: {source.get('metadata', {})}")
            else:
                print("   ❌ No sources returned!")
            
            return True
            
        except Exception as e:
            print(f"   ❌ RAG query processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ RAG agents debug failed: {e}")
        return False

async def debug_hyde_service():
    """Debug HYDE service."""
    print("\n" + "=" * 50)
    print("🔬 DEBUGGING HYDE SERVICE")
    print("=" * 50)
    
    try:
        from hyde_service import hyde_service
        
        test_query = "Ada berapa ayat dari UUD 1945 Pasal 32?"
        print(f"\nTesting HYDE for query: '{test_query}'")
        
        if settings.hyde_enabled:
            hyde_result = await hyde_service.generate_hypothetical_documents(test_query)
            if hyde_result:
                print(f"   ✅ HYDE generated successfully")
                print(f"   📝 Enhanced query: {hyde_result.enhanced_query[:200]}...")
                print(f"   📄 Hypothetical docs: {len(hyde_result.hypothetical_documents)}")
            else:
                print("   ❌ HYDE generation failed")
        else:
            print("   ⚠️ HYDE is disabled in settings")
            
    except Exception as e:
        print(f"❌ HYDE debug failed: {e}")

async def debug_system_prompts():
    """Debug system prompts."""
    print("\n" + "=" * 50)
    print("📝 DEBUGGING SYSTEM PROMPTS")
    print("=" * 50)
    
    try:
        from config import SYSTEM_PROMPTS
        
        print("\nAvailable system prompts:")
        for key in SYSTEM_PROMPTS.keys():
            prompt = SYSTEM_PROMPTS[key]
            print(f"   {key}: {len(prompt)} characters")
            print(f"   Preview: {prompt[:100]}...")
            
    except Exception as e:
        print(f"❌ System prompts debug failed: {e}")

async def run_comprehensive_debug():
    """Run comprehensive debug of the entire RAG system."""
    print("🚀 STARTING COMPREHENSIVE RAG SYSTEM DEBUG")
    print("=" * 60)
    
    # Debug vector store
    vector_store_ok = await debug_vector_store()
    
    # Debug HYDE service
    await debug_hyde_service()
    
    # Debug system prompts
    await debug_system_prompts()
    
    # Debug RAG agents (only if vector store is OK)
    if vector_store_ok:
        rag_agents_ok = await debug_rag_agents()
    else:
        print("\n⚠️ Skipping RAG agents debug due to vector store issues")
        rag_agents_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 DEBUG SUMMARY")
    print("=" * 60)
    print(f"Vector Store: {'✅ OK' if vector_store_ok else '❌ FAILED'}")
    print(f"RAG Agents: {'✅ OK' if rag_agents_ok else '❌ FAILED'}")
    
    if not vector_store_ok:
        print("\n💡 SOLUTIONS:")
        print("1. Check if documents are properly uploaded")
        print("2. Verify vector store initialization")
        print("3. Check database connection")
        print("4. Verify embedding model is working")
    
    if not rag_agents_ok and vector_store_ok:
        print("\n💡 SOLUTIONS:")
        print("1. Check LLM configuration (Gemini API key)")
        print("2. Verify system prompts")
        print("3. Check workflow configuration")

async def test_specific_search():
    """Test specific search for UUD 1945."""
    print("\n" + "=" * 50)
    print("🎯 TESTING SPECIFIC UUD 1945 SEARCH")
    print("=" * 50)
    
    # Comprehensive search terms for UUD 1945 Pasal 32
    search_terms = [
        "UUD 1945 Pasal 32",
        "Undang-Undang Dasar 1945 Pasal 32",
        "Pasal 32 UUD 1945",
        "kebudayaan nasional Indonesia",
        "identitas bangsa Indonesia",
        "Pasal 32",
        "kebudayaan",
        "UUD 1945"
    ]
    
    for term in search_terms:
        print(f"\n🔍 Searching for: '{term}'")
        try:
            # Try the fixed search method from our RAG agents
            if hasattr(vector_store_service, 'similarity_search'):
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: vector_store_service.similarity_search(term, k=3)
                )
            else:
                print("   ❌ No search method available")
                continue
                
            print(f"   Found {len(results)} results")
            
            for i, result in enumerate(results):
                print(f"\n   Result {i+1}:")
                content = ""
                metadata = {}
                
                if hasattr(result, 'page_content'):
                    content = result.page_content
                    metadata = getattr(result, 'metadata', {})
                elif hasattr(result, 'content'):
                    content = result.content
                    metadata = getattr(result, 'metadata', {})
                elif isinstance(result, dict):
                    content = result.get('content', result.get('page_content', ''))
                    metadata = result.get('metadata', {})
                
                print(f"     Content preview: {content[:150]}...")
                print(f"     Metadata: {metadata}")
                print(f"     Content length: {len(content)} chars")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

if __name__ == "__main__":
    async def main():
        await run_comprehensive_debug()
        await test_specific_search()
    
    asyncio.run(main())