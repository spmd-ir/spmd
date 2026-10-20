// RUN: spmd-opt %s -convert-scf-switch-to-if | FileCheck %s --implicit-check-not=scf.index_switch
//
// scf.index_switch becomes a chain of scf.if on `arg == case`, one per case in
// order, with the default in the innermost else. spmd-verify does not run this
// pass. A switch with results is not tested: they are replaced by what the
// default yields (MR 16's description, finding 19).

// CHECK-LABEL: func.func @dispatch(
// CHECK-SAME:    %[[I:.*]]: index, %[[X:.*]]: i32)
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK:         %[[IS1:.*]] = arith.cmpi eq, %[[I]], %[[C1]] : index
// CHECK-NEXT:    scf.if %[[IS1]] {
// CHECK-NEXT:      func.call @one(%[[X]])
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[IS5:.*]] = arith.cmpi eq, %[[I]], %[[C5]] : index
// CHECK-NEXT:      scf.if %[[IS5]] {
// CHECK-NEXT:        func.call @five(%[[X]])
// CHECK-NEXT:      } else {
// CHECK-NEXT:        func.call @other(%[[X]])
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func @dispatch(%i: index, %x: i32) {
  scf.index_switch %i
  case 1 {
    func.call @one(%x) : (i32) -> ()
    scf.yield
  }
  case 5 {
    func.call @five(%x) : (i32) -> ()
    scf.yield
  }
  default {
    func.call @other(%x) : (i32) -> ()
    scf.yield
  }
  return
}
func.func private @one(i32)
func.func private @five(i32)
func.func private @other(i32)
