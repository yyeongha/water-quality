//paint_expert
var sumInit = function(){

};


sumInit.prototype = {
	init : function(){
		console.log("init")
		var owner = this;
		this.node = $(".paint_cost > ul > li");

		this.node.find(".calc_btn > div > a").on("click", function(e){
			e.preventDefault();

			owner.sumSet($(this).parent().index(), $(this).parents(".on").index());
		});

		$(".paint_cost input:text").blur(function(){
			if(!$.isNumeric($(this).val()) && $(this).val().length > 0) {
				owner.sumSet(1);

				alert("숫자만 입력 가능합니다.");

				$(this).focus();

				return false;
			};
		});
	},

	sumSet : function(num, nodeNum){
		if(num == 0){
			//add sum
			var numArr = [];
			var total = 0;

			this.node.eq(nodeNum).find(".input_calc > ul > li").each(function(){
				numArr.push($(this).find("input").val());
			});

			switch(nodeNum){
				case 0: //이론도포면적
					total = (10 * numArr[0]) / numArr[1];

				break;
				
				case 1: //이론 도료소요량
					total = ((10 * numArr[0]) / numArr[1]) * numArr[2];
				
				break;

				case 2: //이론 도료가격
					total = ((10 * numArr[0]) / numArr[1]) * numArr[2] * numArr[3];
				
				break;

				case 3: //실제도포면적
					total = (numArr[0] * (100 - numArr[1])) / 100;
				
				break;

				case 4: //실제 도료소요량
					total = ((numArr[0] * 100) / (100 - numArr[1])) * ((numArr[2] * numArr[3]) - (numArr[4] * numArr[5]));
				
				break;

				case 5: //건조도막두께
					total = (numArr[0] * numArr[1]) / 100;
				
				break;

				case 6: //젖은도막두께
					total = (numArr[0] * 100) / numArr[1];
				
				break;

				case 7: //단위부피당 건조 도막 무게
					total = (numArr[0] * numArr[1]) / numArr[2];
				
				break;

				case 8: //단위면적당 건조 도막 무게
					total = (numArr[0] * numArr[1] * numArr[2]) / (1000 * numArr[3]);
				
				break;

				case 9: //온도환산 섭씨
					total = (5 / 9) * (numArr[0] - 32);
				
				break;

				case 10: //온도환산 화씨
					total = (9 / 5) * numArr[0] + 32;
				
				break;

				case 11: //부식지수
					total = ((numArr[0] - 65) / 10) * 1.054 * numArr[1];
				
				break;

				case 12: //예상 부식량
					total = 4.153 + 0.880 * numArr[0] - 0.073 * numArr[1] - 0.032 * numArr[2] + 2.913 * numArr[3] + 4.721 * numArr[4];
				
				break;

				case 13: //예상 부식
					total = numArr[0] * 0.0365 / 7.86;
				
				break;

				case 14: //㎡를 예전 평수로
					total = (121 / 400) * numArr[0];
				
				break;

				case 15: //예전 평수를 ㎡으로
					total = (121 / 400) * numArr[0];
				
				break;
			};

			total = Math.round(total);

			this.node.eq(nodeNum).find(".calc_result span").text(total);
		}else{
			//Initialize
			$("input").val("");
			this.node.eq(nodeNum).find(".calc_result span").text("0");
		};
	}
};

//diy_product 
var simpleSum = {
	init : function(){
		var owner = this;
		var str = this.selectProduct();

		$(".select").change(function () {
			str = "";
			str = owner.selectProduct();
		});

		$(".paint_cal .btn_space .btn_blue").on("click", function(e){
			e.preventDefault();

			owner.sumSet(str);
		});

		$(".paint_cal .btn_space .btn_gray").on("click", function(e){
			e.preventDefault();

			owner.firstSet();
		});
		
		$(".paint_cal input:text").blur(function(){
			if(!$.isNumeric($(this).val()) && $(this).val().length > 0) {
				owner.firstSet();

				alert("숫자만 입력 가능합니다.");
				
				$(this).focus();

				return false;
			};
		});
	},

	selectProduct : function(){
		var str;
		$(".select option:selected").each(function() {
			str = $(this).index();
		});

		this.firstSet();
		
		return str;
	},

	sumSet : function(chk){
		var area;
		var total;

		switch(chk){
			case 0: //드림코트
				total = $(".paint_cal .cal_input .input_area .horizon").val() * $(".paint_cal .cal_input .input_area .vertical").val() * 0.15;

			break;
			
			case 1: //아쿠아우드 스테인
				total = $(".paint_cal .cal_input .input_area .horizon").val() * $(".paint_cal .cal_input .input_area .vertical").val() * 0.125;
			
			break;

			case 2: //오일 스테인
				total = $(".paint_cal .cal_input .input_area .horizon").val() * $(".paint_cal .cal_input .input_area .vertical").val() * 0.17;
			
			break;
		};

		area = $(".paint_cal .cal_input .input_area .horizon").val() * $(".paint_cal .cal_input .input_area .vertical").val();

		area = Math.round(area);
		total = Math.round(total);

		$(".diy_result > dl:eq(0) dd span").text(area);
		$(".diy_result > dl:eq(1) dd span").text(total);
	},

	firstSet : function(){
		$(".diy_result > dl:eq(0) dd span").text(0);
		$(".diy_result > dl:eq(1) dd span").text(0);

		$(":text").val("").removeClass("off");
	}
};