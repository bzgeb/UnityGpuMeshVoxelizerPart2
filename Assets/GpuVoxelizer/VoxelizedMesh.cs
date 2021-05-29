#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.Profiling;

[ExecuteAlways]
public class VoxelizedMesh : MonoBehaviour
{
    [SerializeField] MeshFilter _meshFilter;
    [SerializeField] MeshCollider _meshCollider;
    [SerializeField] float _halfSize = 0.05f;
    [SerializeField] Vector3 _boundsMin;

    [SerializeField] Material _gridPointMaterial;
    [SerializeField] int _gridPointCount;

    [SerializeField] ComputeShader _voxelizeComputeShader;
    ComputeBuffer _voxelPointsBuffer;
    ComputeBuffer _meshVerticesBuffer;
    ComputeBuffer _meshTrianglesBuffer;

    ComputeBuffer _pointsArgsBuffer;

    [SerializeField] bool _drawDebug;

    static readonly int LocalToWorldMatrix = Shader.PropertyToID("_LocalToWorldMatrix");
    static readonly int BoundsMin = Shader.PropertyToID("_BoundsMin");
    static readonly int VoxelGridPoints = Shader.PropertyToID("_VoxelGridPoints");

    Vector4[] _gridPoints;

    void OnEnable()
    {
        _pointsArgsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
    }

    void OnDisable()
    {
        _pointsArgsBuffer?.Dispose();
        _voxelPointsBuffer?.Dispose();
        _meshTrianglesBuffer?.Dispose();
        _meshVerticesBuffer?.Dispose();
    }

    void Update()
    {
        VoxelizeMeshWithGPU();

        if (_drawDebug)
        {
            _gridPointMaterial.SetMatrix(LocalToWorldMatrix, transform.localToWorldMatrix);
            _gridPointMaterial.SetVector(BoundsMin, new Vector4(_boundsMin.x, _boundsMin.y, _boundsMin.z, 0.0f));
            _gridPointMaterial.SetBuffer(VoxelGridPoints, _voxelPointsBuffer);
            _pointsArgsBuffer.SetData(new[] {1, _gridPointCount, 0, 0, 0});
            Graphics.DrawProceduralIndirect(_gridPointMaterial, _meshCollider.bounds, MeshTopology.Points,
                _pointsArgsBuffer);
        }
    }

    void VoxelizeMeshWithGPU()
    {
        Profiler.BeginSample("Voxelize Mesh (GPU)");

        Bounds bounds = _meshCollider.bounds;
        _boundsMin = transform.InverseTransformPoint(bounds.min);

        Vector3 voxelCount = bounds.extents / _halfSize;
        int xGridSize = Mathf.CeilToInt(voxelCount.x);
        int yGridSize = Mathf.CeilToInt(voxelCount.y);
        int zGridSize = Mathf.CeilToInt(voxelCount.z);

        bool resizeVoxelPointsBuffer = false;
        if (_gridPoints == null || _gridPoints.Length != xGridSize * yGridSize * zGridSize ||
            _voxelPointsBuffer == null)
        {
            _gridPoints = new Vector4[xGridSize * yGridSize * zGridSize];
            resizeVoxelPointsBuffer = true;
        }

        if (resizeVoxelPointsBuffer || _voxelPointsBuffer == null || !_voxelPointsBuffer.IsValid())
        {
            _voxelPointsBuffer?.Dispose();
            _voxelPointsBuffer = new ComputeBuffer(xGridSize * yGridSize * zGridSize, 4 * sizeof(float));
        }

        if (resizeVoxelPointsBuffer)
        {
            _voxelPointsBuffer.SetData(_gridPoints);
        }

        if (_meshVerticesBuffer == null || !_meshVerticesBuffer.IsValid())
        {
            _meshVerticesBuffer?.Dispose();
            
            var sharedMesh = _meshFilter.sharedMesh;
            _meshVerticesBuffer = new ComputeBuffer(sharedMesh.vertexCount, 3 * sizeof(float));
            _meshVerticesBuffer.SetData(sharedMesh.vertices);
        }

        if (_meshTrianglesBuffer == null || !_meshTrianglesBuffer.IsValid())
        {
            _meshTrianglesBuffer?.Dispose();

            var sharedMesh = _meshFilter.sharedMesh;
            _meshTrianglesBuffer = new ComputeBuffer(sharedMesh.triangles.Length, sizeof(int));
            _meshTrianglesBuffer.SetData(sharedMesh.triangles);
        }

        var voxelizeKernel = _voxelizeComputeShader.FindKernel("VoxelizeMesh");
        _voxelizeComputeShader.SetInt("_GridWidth", xGridSize);
        _voxelizeComputeShader.SetInt("_GridHeight", yGridSize);
        _voxelizeComputeShader.SetInt("_GridDepth", zGridSize);

        _voxelizeComputeShader.SetFloat("_CellHalfSize", _halfSize);

        _voxelizeComputeShader.SetBuffer(voxelizeKernel, VoxelGridPoints, _voxelPointsBuffer);
        _voxelizeComputeShader.SetBuffer(voxelizeKernel, "_MeshVertices", _meshVerticesBuffer);
        _voxelizeComputeShader.SetBuffer(voxelizeKernel, "_MeshTriangleIndices", _meshTrianglesBuffer);
        _voxelizeComputeShader.SetInt("_TriangleCount", _meshFilter.sharedMesh.triangles.Length);

        _voxelizeComputeShader.SetVector(BoundsMin, _boundsMin);

        _voxelizeComputeShader.GetKernelThreadGroupSizes(voxelizeKernel, out uint xGroupSize, out uint yGroupSize,
            out uint zGroupSize);

        _voxelizeComputeShader.Dispatch(voxelizeKernel,
            Mathf.CeilToInt(xGridSize / (float) xGroupSize),
            Mathf.CeilToInt(yGridSize / (float) yGroupSize),
            Mathf.CeilToInt(zGridSize / (float) zGroupSize));
        _gridPointCount = _voxelPointsBuffer.count;

        Profiler.EndSample();
    }

#if UNITY_EDITOR
    void Reset()
    {
        _meshFilter = GetComponent<MeshFilter>();
        if (TryGetComponent(out MeshCollider meshCollider))
        {
            _meshCollider = meshCollider;
        }
        else
        {
            _meshCollider = gameObject.AddComponent<MeshCollider>();
        }

        var basePath = "Assets/GpuVoxelizer/";
        _gridPointMaterial = AssetDatabase.LoadAssetAtPath<Material>($"{basePath}Materials/GridPointMaterial.mat");
        _voxelizeComputeShader =
            AssetDatabase.LoadAssetAtPath<ComputeShader>($"{basePath}ComputeShaders/VoxelizeMesh.compute");
    }
#endif
}