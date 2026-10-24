// Offline SE(3) pose graph using Ceres AutoDiff and quaternion manifolds.
// Poses T_wc; edge measurement inv(T_wc_source) * T_wc_target.
// No images, ground truth, frontend state, or online entry points are accessed.
// V33: an explicit loss mode; original legacy graph source stays unchanged.
#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
struct Pose { Eigen::Vector3d p; Eigen::Quaterniond q; };
struct Edge { int a,b,aux; Pose z; double st,sr; };
struct RelativeCost {
 Edge e;
 template<typename T> bool operator()(const T* pa,const T* qa,const T* pb,const T* qb,T* r) const {
  Eigen::Map<const Eigen::Matrix<T,3,1>> a(pa),b(pb);
  Eigen::Map<const Eigen::Quaternion<T>> ra(qa),rb(qb);
  Eigen::Quaternion<T> predicted=ra.conjugate()*rb;
  Eigen::Quaternion<T> error=e.z.q.cast<T>()*predicted.conjugate();
  // q and -q are the same rotation; use the shortest chord.
  if(error.w()<T(0))error.coeffs()=-error.coeffs();
  Eigen::Map<Eigen::Matrix<T,3,1>> rt(r),rr(r+3);
  rt=(ra.conjugate()*(b-a)-e.z.p.cast<T>())/T(e.st);
  rr=T(2)*error.vec()/T(e.sr);
  return true;
 }
};
void expect(std::istream& f,const std::string& wanted) {
 std::string v;if(!(f>>v)||v!=wanted)throw std::runtime_error("invalid graph token "+wanted);
}
Pose readPose(std::istream& f) {
 Pose p;double x,y,z,w;
 if(!(f>>p.p.x()>>p.p.y()>>p.p.z()>>x>>y>>z>>w))throw std::runtime_error("truncated pose");
 p.q=Eigen::Quaterniond(w,x,y,z);
 if(!p.p.allFinite()||!p.q.coeffs().allFinite()||std::abs(p.q.norm()-1)>1e-6)throw std::runtime_error("invalid pose");
 p.q.normalize();return p;
}
int main(int argc,char** argv) {
 try {
  if(argc!=4)throw std::runtime_error("usage: offline_confidence_pose_graph INPUT NEW_OUTPUT legacy|quadratic");
  const std::string mode=argv[3];
  if(mode!="legacy" && mode!="quadratic")throw std::runtime_error("invalid loss mode");
  std::filesystem::path out(argv[2]);if(std::filesystem::exists(out))throw std::runtime_error("refuse overwrite");
  std::ifstream f(argv[1]);expect(f,"AFS_KF_GRAPH");int v,n,m;f>>v;if(v!=1)throw std::runtime_error("bad version");
  expect(f,"NODES");f>>n;if(n<2||n>10000)throw std::runtime_error("node budget");
  std::vector<Pose> poses(n);
  for(int i=0;i<n;++i){expect(f,"V");int id;f>>id;if(id!=i)throw std::runtime_error("node order");poses[i]=readPose(f);}
  auto before=poses;expect(f,"EDGES");f>>m;if(m<n-1||m>2*(n-1))throw std::runtime_error("edge budget");
  std::vector<Edge> edges;
  for(int i=0;i<m;++i){
   expect(f,"E");Edge e;f>>e.a>>e.b>>e.aux;
   if(e.a<0||e.b!=e.a+1||e.b>=n||(e.aux!=0&&e.aux!=1))throw std::runtime_error("invalid edge");
   e.z=readPose(f);f>>e.st>>e.sr;
   if(!std::isfinite(e.st)||!std::isfinite(e.sr)||std::min(e.st,e.sr)<=0)throw std::runtime_error("invalid scales");
   edges.push_back(e);
  }
  expect(f,"END");std::string extra;if(f>>extra)throw std::runtime_error("trailing graph");
  std::filesystem::create_directory(out);
  ceres::Problem problem;
  for(auto& p:poses){problem.AddParameterBlock(p.p.data(),3);problem.AddParameterBlock(p.q.coeffs().data(),4,new ceres::EigenQuaternionManifold);}
  for(const auto& e:edges){
   auto* c=new ceres::AutoDiffCostFunction<RelativeCost,6,3,4,3,4>(new RelativeCost{e});
   problem.AddResidualBlock(c,e.aux && mode=="legacy"?new ceres::HuberLoss(1.):nullptr,poses[e.a].p.data(),poses[e.a].q.coeffs().data(),poses[e.b].p.data(),poses[e.b].q.coeffs().data());
  }
  problem.SetParameterBlockConstant(poses[0].p.data());problem.SetParameterBlockConstant(poses[0].q.coeffs().data());
  ceres::Solver::Options options;options.max_num_iterations=50;options.num_threads=1;options.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  std::ofstream report(out/"report.json");report<<std::setprecision(17)<<"{\n\"usable\":"<<(summary.IsSolutionUsable()?"true":"false")<<",\n\"initial_cost\":"<<summary.initial_cost<<",\n\"final_cost\":"<<summary.final_cost<<",\n\"solve_s\":"<<summary.total_time_in_seconds<<",\n\"iterations\":"<<summary.iterations.size()<<",\n\"termination_type\":"<<int(summary.termination_type)<<"\n}\n";
  std::ofstream nodes(out/"nodes.txt");nodes<<std::setprecision(17);
  for(int i=0;i<n;++i){const auto& p=poses[i];nodes<<i<<" "<<p.p.transpose()<<" "<<p.q.x()<<" "<<p.q.y()<<" "<<p.q.z()<<" "<<p.q.w()<<"\n";}
  std::ofstream residuals(out/"residuals.csv");residuals<<std::setprecision(17)<<"source,target,auxiliary,before_whitened_norm,after_whitened_norm\n";
  for(const auto& e:edges){
   double a[6],b[6];RelativeCost cost{e};
   cost(before[e.a].p.data(),before[e.a].q.coeffs().data(),before[e.b].p.data(),before[e.b].q.coeffs().data(),a);
   cost(poses[e.a].p.data(),poses[e.a].q.coeffs().data(),poses[e.b].p.data(),poses[e.b].q.coeffs().data(),b);
   residuals<<e.a<<","<<e.b<<","<<e.aux<<","<<Eigen::Map<Eigen::Matrix<double,6,1>>(a).norm()<<","<<Eigen::Map<Eigen::Matrix<double,6,1>>(b).norm()<<"\n";
  }
  std::cout<<summary.BriefReport()<<"\n";
  return summary.IsSolutionUsable()?0:3;
 } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
