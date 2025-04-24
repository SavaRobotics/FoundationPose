import os
import sys
import traceback

def test_pytorch_imports():
    """Test PyTorch and PyTorch3D imports and availability."""
    results = []
    
    # Check Python version
    results.append(f"Python version: {sys.version}")
    
    # Test basic PyTorch
    try:
        import torch
        results.append(f"PyTorch version: {torch.__version__}")
        results.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            results.append(f"CUDA version: {torch.version.cuda}")
            results.append(f"GPU count: {torch.cuda.device_count()}")
            results.append(f"Current device: {torch.cuda.current_device()}")
            results.append(f"Device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        results.append(f"PyTorch error: {str(e)}")
        results.append(traceback.format_exc())
    
    # Test PyTorch3D
    try:
        import pytorch3d
        results.append(f"PyTorch3D version: {pytorch3d.__version__}")
        
        # Check for renderer module
        try:
            import pytorch3d.renderer
            results.append("pytorch3d.renderer imported successfully")
            
            # List available submodules in renderer
            renderer_dir = dir(pytorch3d.renderer)
            renderer_modules = [m for m in renderer_dir if not m.startswith('_')]
            results.append(f"Available renderer modules: {renderer_modules}")
            
            # Specifically check for dibr
            if 'dibr' in renderer_modules:
                results.append("dibr module found")
                
                # Try the specific import that's failing
                try:
                    import pytorch3d.renderer.dibr as dr
                    results.append("pytorch3d.renderer.dibr imported successfully")
                    
                    # Check if RasterizeCudaContext exists
                    if hasattr(dr, 'RasterizeCudaContext'):
                        results.append("RasterizeCudaContext found in dibr module")
                        
                        # Try creating an instance
                        try:
                            ctx = dr.RasterizeCudaContext()
                            results.append("Successfully created RasterizeCudaContext")
                        except Exception as e:
                            results.append(f"Error creating RasterizeCudaContext: {str(e)}")
                    else:
                        results.append("RasterizeCudaContext NOT found in dibr module")
                        results.append(f"Available in dibr: {dir(dr)}")
                except Exception as e:
                    results.append(f"Error importing pytorch3d.renderer.dibr: {str(e)}")
            else:
                results.append("dibr module NOT found")
                
                # Check alternative import paths
                try:
                    from pytorch3d.renderer import RasterizeCudaContext
                    results.append("Found RasterizeCudaContext directly in pytorch3d.renderer")
                except ImportError:
                    results.append("RasterizeCudaContext not found in pytorch3d.renderer")
                
                try:
                    from pytorch3d.renderer import rasterize_meshes
                    results.append("Found rasterize_meshes in pytorch3d.renderer")
                except ImportError:
                    results.append("rasterize_meshes not found in pytorch3d.renderer")
                    
        except Exception as e:
            results.append(f"Error importing pytorch3d.renderer: {str(e)}")
    except Exception as e:
        results.append(f"PyTorch3D error: {str(e)}")
        results.append(traceback.format_exc())
    
    # Print installation paths
    try:
        import pytorch3d
        results.append(f"PyTorch3D installed at: {os.path.dirname(pytorch3d.__file__)}")
    except:
        pass
    
    return results

if __name__ == "__main__":
    results = test_pytorch_imports()
    print("\n".join(results))
    
    # Save results to file
    with open("pytorch3d_test_results.txt", "w") as f:
        f.write("\n".join(results))
    print("\nResults saved to pytorch3d_test_results.txt")